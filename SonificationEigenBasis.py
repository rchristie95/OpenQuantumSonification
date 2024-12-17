globals().clear()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.special import factorial, eval_hermite
from scipy.linalg import eig, expm, kron
import soundfile as sf
from numpy.fft import fft, ifft, fftshift, ifftshift
import logging
import os
import subprocess

# For animations, one could use matplotlib.animation, but here we will just store frames
# from matplotlib.animation import FFMpegWriter


def WignerT(Ex, x, p, hbar):
    """
    Python equivalent of the MATLAB function W = WignerT(Ex, x, p, hbar).

    Parameters
    ----------
    Ex : ndarray
        Wavefunction in position representation, shape (Nx, N).
    x, p : ndarray
        Position and momentum arrays, each length Nx.
    hbar : float
        Reduced Planck's constant.

    Returns
    -------
    W : ndarray
        Wigner function array of shape (Nx, Nx, N).
    """
    N = Ex.shape[1]
    Nx = Ex.shape[0]

    # In MATLAB code:
    # x=x.'; p=p.'; makes x and p row vectors.
    # We just ensure they are 1D arrays here.
    x = x.flatten()
    p = p.flatten()

    # x = ifftshift(x); In MATLAB this shifts the vector x.
    # Usually, the Wigner transform is defined for symmetrical arrays.
    # If needed, you can consider whether shifting x is required or not.
    # We'll follow the code literally:
    x = ifftshift(x)

    # Prepare output
    W = np.zeros((Nx, Nx, N), dtype=float)

    # Precompute Fourier transforms of Ex columns
    # We'll follow the loop structure as in MATLAB
    for n in range(N):
        # fft of Ex[:,n]
        fft_Ex_n = fft(Ex[:, n])

        # Create matrices by outer products
        # exp(1i*x*p'/2/hbar) -> exp(1j*(x[:,None]*p[None,:])/(2*hbar))
        phase_pos = np.exp((1j * np.outer(x, p)) / (2*hbar))
        phase_neg = np.exp((-1j * np.outer(x, p)) / (2*hbar))

        # Repeat fft_Ex_n along columns (Nx x Nx)
        fft_Ex_n_matrix = np.tile(fft_Ex_n[:, np.newaxis], (1, Nx))

        # EX1 and EX2 computations
        EX1 = ifft(fft_Ex_n_matrix * phase_pos, axis=0)
        EX2 = ifft(fft_Ex_n_matrix * phase_neg, axis=0)

        # Compute Wigner function
        # W(:,:,n) = (1/2/pi/hbar)*real(fftshift(fft(fftshift(EX1.*conj(EX2),2),[],2),2))'
        temp = EX1 * np.conj(EX2)
        # fftshift along axis=1 (columns)
        temp_shift1 = fftshift(temp, axes=1)
        temp_fft = fft(temp_shift1, axis=1)
        temp_shift2 = fftshift(temp_fft, axes=1)

        # Take real part, scale, and transpose
        W_slice = (1/(2*np.pi*hbar))*np.real(temp_shift2).T
        W[:,:,n] = W_slice

    return W


def HamiltonianExponentiate(PsiIn, N, tf, Hhat, hbar):
    """
    Python equivalent of:
    function [outPsi,outRho]= HamiltonianExponentiate(PsiIn,dt,tf,Hhat,hbar)

    Evolves the initial state PsiIn under the Hamiltonian Hhat for time tf with step dt.
    """
    # The MATLAB code fixes N=2000 and dt=tf/N.
    # Here, we already have dt and tf, so we can deduce N if needed.
    # Or just trust that the provided dt and tf are consistent.

    dt=tf/N
    bSize = Hhat.shape[0]

    Psi = np.zeros((bSize, N), dtype=complex)
    Psi[:,0] = PsiIn

    # Propagator
    UProp = expm(-1j*Hhat*dt/hbar)

    outRho = np.zeros((bSize,bSize,N), dtype=complex)
    outRho[:,:,0] = np.outer(PsiIn, PsiIn.conjugate())

    for n in range(1,N):
        Psi[:,n] = UProp @ Psi[:,n-1]
        Psi[:,n] = Psi[:,n]/np.linalg.norm(Psi[:,n])
        outRho[:,:,n] = np.outer(Psi[:,n], Psi[:,n].conjugate())

    outPsi = Psi
    return outPsi, outRho


def HamiltonianEigenrep(RhoIn, InvVectors, N):
    """
    Python equivalent of:
    function [outRho]= HamiltonianEigenrep(RhoIn,InvVectors,N)

    Transforms the density matrices RhoIn into the eigenbasis given by InvVectors.
    """
    bSize = RhoIn.shape[0]
    outRho = np.zeros((bSize,bSize,N), dtype=complex)

    for n in range(N):
        TempRho = InvVectors @ RhoIn[:,:,n] @ InvVectors.conjugate().T
        # Symmetrize
        TempRho = 0.5*(TempRho + TempRho.conjugate().T)
        # Normalize
        tr = np.trace(TempRho)
        if tr != 0:
            TempRho = TempRho/tr
        outRho[:,:,n] = TempRho

    return outRho
def SSEDynamicsSRK(PsiIn: np.ndarray, dt: float, tf: float, 
                  Hhat: np.ndarray, Lhat: np.ndarray, 
                  hbar: float, eta: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Simulate the Stochastic Schrödinger Equation dynamics using the SRK method.

    Parameters:
    - PsiIn (np.ndarray): Initial state vector (shape: (bSize,))
    - dt (float): Time step
    - tf (float): Final time
    - Hhat (np.ndarray): Hamiltonian matrix (shape: (bSize, bSize))
    - Lhat (np.ndarray): Lindblad operator matrix (shape: (bSize, bSize))
    - hbar (float): Reduced Planck constant
    - eta (np.ndarray): Noise array (shape: (N,))

    Returns:
    - PsiOut (np.ndarray): Time-evolved state vectors (shape: (bSize, num_steps))
    - RhoOut (np.ndarray): Time-evolved density matrices (shape: (bSize, bSize, num_steps))
    """
    # Total number of time steps
    N = int(round(tf / dt))
    
    # Basis size
    bSize = len(PsiIn)
    
    # Initial state
    Psi0 = PsiIn.copy()
    
    # Precompute square root of dt
    Sqdt = np.sqrt(dt)
    
    # Preallocate output arrays
    num_steps = 2000  # As per MATLAB's PsiOut size
    PsiOut = np.zeros((bSize, num_steps), dtype=complex)
    RhoOut = np.zeros((bSize, bSize, num_steps), dtype=complex)
    
    # Initialize the first column
    PsiOut[:, 0] = PsiIn
    RhoOut[:, :, 0] = np.outer(PsiIn, PsiIn.conj())
    
    # Determine the frequency of saving data
    modfactor = N / 2000
    if modfactor < 1:
        modfactor = 1  # Ensure at least every step is saved if N < 2000
    
    # Initialize step counter
    m = 1  # Python uses 0-based indexing; m=1 corresponds to the second column
    
    for n in range(2, N + 1):
        # Real noise
        eta1 = eta[n - 1]  # Adjust for 0-based indexing
        etaSq1 = 0.5 * (eta1**2 - 1)
        
        # Calculate drift and stochastic components
        DPsi0 = Drift(Psi0, Hhat, Lhat, bSize, hbar)
        SPsi0 = Stoch(Psi0, Lhat, bSize, hbar)
        
        # Intermediate Psi calculations
        Psi1 = Psi0 + DPsi0 * dt
        Psi2 = Psi0 + DPsi0 * dt + SPsi0 * Sqdt * etaSq1
        Psi3 = Psi0 + DPsi0 * dt - SPsi0 * Sqdt * etaSq1
        
        # Calculate intermediate terms for SRK
        T1 = Psi0 + 0.5 * (DPsi0 + Drift(Psi1, Hhat, Lhat, bSize, hbar)) * dt
        T2 = SPsi0 * eta1 * Sqdt
        T3 = 0.5 * Sqdt * (Stoch(Psi2, Lhat, bSize, hbar) - Stoch(Psi3, Lhat, bSize, hbar))
        
        # Update Psi0 using SRK method
        Psi0 = T1 + T2 + T3
        
        # Normalize Psi0
        Psi0 = Psi0 / np.linalg.norm(Psi0)
        
        # Save the state at specified intervals
        if n % modfactor == 0:
            if m < num_steps:
                PsiOut[:, m] = Psi0
                RhoOut[:, :, m] = np.outer(Psi0, Psi0.conj())
                m += 1
            else:
                # Prevent index out of range if m exceeds num_steps
                break
    
    # Trim the preallocated arrays to actual saved steps
    PsiOut = PsiOut[:, :m]
    RhoOut = RhoOut[:, :, :m]
    
    return PsiOut, RhoOut

def Drift(Psi: np.ndarray, H: np.ndarray, L: np.ndarray, bSize: int, hbar: float) -> np.ndarray:
    """
    Calculate the deterministic drift component of the dynamics.

    Parameters:
    - Psi (np.ndarray): Current state vector (shape: (bSize,))
    - H (np.ndarray): Hamiltonian matrix (shape: (bSize, bSize))
    - L (np.ndarray): Lindblad operator matrix (shape: (bSize, bSize))
    - bSize (int): Size of the basis
    - hbar (float): Reduced Planck constant

    Returns:
    - np.ndarray: Drift component (shape: (bSize,))
    """
    # Calculate expectation value: <Psi|L|Psi>
    EPsi = np.dot(Psi.conj().T, np.dot(L, Psi))
    
    # Compute the drift term
    drift = ( (-1j * H - 0.5 * (L.conj().T @ L) + EPsi.conj() * L - 
               0.5 * (EPsi * EPsi.conj()) * np.eye(bSize) ) @ Psi ) / hbar
    
    return drift


def Stoch(Psi: np.ndarray, L: np.ndarray, bSize: int, hbar: float) -> np.ndarray:
    """
    Calculate the stochastic component of the dynamics.

    Parameters:
    - Psi (np.ndarray): Current state vector (shape: (bSize,))
    - L (np.ndarray): Lindblad operator matrix (shape: (bSize, bSize))
    - bSize (int): Size of the basis
    - hbar (float): Reduced Planck constant

    Returns:
    - np.ndarray: Stochastic component (shape: (bSize,))
    """
    # Calculate expectation value: <Psi|L|Psi>
    EPsi = np.dot(Psi.conj().T, np.dot(L, Psi))
    
    # Compute the stochastic term
    stoch = ( L - EPsi * np.eye(bSize) ) @ Psi * np.sqrt(1 / hbar)
    
    return stoch

def rho_wigner(Rho, x, p, hbar):
    """
    Compute the Wigner function for a given density matrix.

    Parameters:
    - Rho: numpy.ndarray
        Density matrix with shape (Nx, Nx, N), where Nx is the spatial dimension and N is the number of time steps.
    - x: numpy.ndarray
        Position grid points with shape (Nx,).
    - p: numpy.ndarray
        Momentum grid points with shape (Nx,).
    - hbar: float
        Reduced Planck constant.

    Returns:
    - W: numpy.ndarray
        Wigner function with shape (Nx, Nx, N).
    """
    N = Rho.shape[2]
    Nx = Rho.shape[0]
    x = x.flatten()
    p = p.flatten()

    # Apply ifftshift to x
    x_shifted = ifftshift(x)

    # Initialize Wigner function array
    W = np.zeros((Nx, Nx, N))

    for n in range(N):
        # Eigendecomposition
        D, V = eig(Rho[:, :, n])
        
        # Initialize Wigner function for this time step
        W_n = np.zeros((Nx, Nx))
        
        for m in range(Nx):
            # Extract eigenvalue and eigenvector
            eigenvalue = D[m]
            eigenvector = V[:, m]
            
            # Compute FFT of the eigenvector
            fft_Vm = fft(eigenvector)
            
            # Compute exponential factors
            exp_pos = np.exp(1j * np.outer(x_shifted, p) / (2 * hbar))
            exp_neg = np.exp(-1j * np.outer(x_shifted, p) / (2 * hbar))
            
            # Compute EX1 and EX2
            EX1 = ifft(fft_Vm[:, np.newaxis] * exp_pos, axis=0)
            EX2 = ifft(fft_Vm[:, np.newaxis] * exp_neg, axis=0)
            
            # Compute the contribution to the Wigner function
            W_m = np.real(
                eigenvalue * (1 / (2 * np.pi * hbar)) *
                fftshift(fft(fftshift(EX1 * np.conj(EX2), axes=1), axis=1), axes=1)
            ).T  # Transpose to match MATLAB's column-major order
            
            # Accumulate the contributions
            W_n += W_m
        
        # Assign the computed Wigner function for this time step
        W[:, :, n] = W_n

        # Optional: Print progress
        if (n + 1) % 100 == 0 or n == N - 1:
            print(f"Computed Wigner function for time step {n + 1}/{N}")

    return W

def lindblad_exp(RhoIn, tf, Hhat, Lhat, hbar, N=2000):
    """
    Compute the Lindblad evolution of a density matrix.

    Parameters:
    - RhoIn: numpy.ndarray
        Initial density matrix with shape (bSize, bSize).
    - tf: float
        Final time.
    - Hhat: numpy.ndarray
        Hamiltonian operator with shape (bSize, bSize).
    - Lhat: numpy.ndarray
        Lindblad operator with shape (bSize, bSize).
    - hbar: float
        Reduced Planck constant.
    - N: int, optional
        Number of time steps. Default is 2000.

    Returns:
    - Rho: numpy.ndarray
        Time-evolved density matrices with shape (bSize, bSize, N).
    """
    bSize = Hhat.shape[0]
    TSpan = np.linspace(0, tf, N)
    dt = TSpan[1] - TSpan[0]

    # Initialize Rho and Vt
    Rho = np.zeros((bSize, bSize, N), dtype=complex)
    Vt = np.zeros((bSize * bSize, N), dtype=complex)
    Id = np.eye(bSize, dtype=complex)

    # Initialize the first density matrix and vector
    Rho[:, :, 0] = RhoIn
    Vt[:, 0] = RhoIn.flatten(order='F')  # MATLAB uses column-major order

    # Define the superoperator M
    # Note: In MATLAB, Hhat.' is the non-conjugate transpose. In Python, Hhat.T is the transpose.
    # If Hhat is complex, use Hhat.conj().T for the conjugate transpose.
    M = (-1j / hbar) * (kron(Id, Hhat) - kron(Hhat.T, Id)) \
        + (1 / hbar) * kron(np.conj(Lhat), Lhat) \
        - (1 / (2 * hbar)) * (
            kron(Id, Lhat.conj().T @ Lhat) + kron((Lhat.conj().T @ Lhat), Id)
        )

    # Compute the propagator
    UProp = expm(M * dt)

    # Time evolution loop
    for n in range(1, N):
        Vt[:, n] = UProp @ Vt[:, n - 1]
        Rho[:, :, n] = Vt[:, n].reshape((bSize, bSize), order='F')  # Reshape in column-major order
        Rho[:, :, n]=Rho[:, :, n]/np.trace(Rho[:, :, n])
        if n % 500 == 0:
            print(f"Progress: {n}/{N} time steps completed.")

    return Rho
#%% ----------------------------------------------
#  Code Starts Here
# ----------------------------------------------

Xview = 4
bSize = 40  # basis size
Nx = 100
hbar = 1
Gamma = 0.1  # lindblad coupling
kT = 0.5

# time stepping
# tf = 40; # defined later in code

# QHO Basis to Position Representation
dX = np.sqrt(2*np.pi/(Nx*hbar))
dP = 2*np.pi/(dX*Nx/hbar)

x = np.zeros(Nx)
p = np.zeros(Nx)
for n in range(Nx):
    x[n] = (-Nx/2+(n))*dX
    p[n] = (-Nx/2+(n))*dP

x1, p1 = np.meshgrid(x, p)

HVector = np.zeros((Nx,bSize), dtype=complex)
# Using eval_hermite from scipy.special
for n in range(bSize):
    # H_n(x) = eval_hermite(n, x)
    Hn = eval_hermite(n, x/np.sqrt(hbar))
    HVector[:,n] = np.sqrt(dX/(2**n * factorial(n))) * (np.pi*hbar)**(-1/4)*np.exp(-x**2/(2*hbar))*Hn

UPos = np.zeros((Nx,Nx), dtype=complex)
for i in range(Nx):
    for j in range(Nx):
        UPos[i,j] = np.sqrt(1/Nx)*np.exp(1j*x[i]*p[j]/hbar)

XhatPos = np.diag(x)
PhatPos = UPos @ np.diag(p) @ UPos.conjugate().T

#%% initial Hamiltonian

# generating operators
a = np.zeros((bSize,bSize), dtype=complex)
for n in range(bSize-1):
    a[n,n+1] = np.sqrt(n+1)  # note indexing shift: n runs 0..bSize-2

a_dag = a.conjugate().T
Xhat = np.sqrt(hbar/2)*(a_dag + a)
Phat = 1j*np.sqrt(hbar/2)*(a_dag - a)

c2 = 0.35  # Quartic
c4 = 0.05

def H(Z):
    return 0.5*Z[1]**2 + c4*Z[0]**4 - c2*Z[0]**2


Hhat = 0.5*Phat@Phat + c4*(Xhat@Xhat@Xhat@Xhat) - c2*(Xhat@Xhat)
HhatGamma = 0.5*Phat@Phat + c4*(Xhat@Xhat@Xhat@Xhat) - c2*(Xhat@Xhat) + 0.5*Gamma*(Xhat@Phat+Phat@Xhat)

Xmin = np.sqrt(c2/(2*c4))
VMin = c4*Xmin**4 - c2*Xmin**2

Lhat = np.sqrt(4*Gamma*kT/hbar)*Xhat + 1j*np.sqrt(Gamma*hbar/(4*kT))*Phat  # caldeira

# Tunnelling graph
Energies, Vectors = np.linalg.eig(Hhat)

# Convert eigenvalues and eigenvectors to real numbers if possible
# This step is optional and depends on whether Hhat is truly Hermitian
logging.basicConfig(level=logging.INFO)

# Sorting eigenvalues in ascending order and rearranging eigenvectors
sorted_indices = np.argsort(Energies)  # Get the indices that would sort the eigenvalues
Energies = Energies[sorted_indices]  # Sorted eigenvalues
Vectors = Vectors[:, sorted_indices]  # Corresponding rearranged eigenvectors


# Normalize eigenvectors (optional but recommended for numerical stability)
Vectors = Vectors / np.linalg.norm(Vectors, axis=0)

# Verify the sorting
logging.info("First 10 sorted eigenvalues:")
logging.info(Energies[:10])

xC = np.linspace(-Xview, Xview, 300)
# Plot potential
plt.figure()
plt.plot(xC, c4*xC**4 - c2*xC**2)
plt.xlim([-Xview, Xview])
for nn in range(7):
    plt.axhline(Energies[nn], color='r', linestyle='--')

Energies = Energies - np.min(Energies) + 1
sortInd = np.argsort(Energies)
MinEnergies = Energies[sortInd[:bSize]]
ind = sortInd[:bSize]

deltaE = MinEnergies[1]-MinEnergies[0]
t_tunnel = np.real(hbar*np.pi/deltaE)
#tf =np.real( 4*t_tunnel)
tf =250
NSSE = 1000000
N = 2000
TSpan = np.linspace(0, tf, N)
InvVectors = np.linalg.inv(Vectors)

#%% Sonification of Arrays
BSound = 12
rootFreq = 60  # in Hz
fs = 44100
duration_per_frame = 1/rootFreq
frame_length_samples = round(duration_per_frame*fs)
total_samples = N*frame_length_samples
total_duration = total_samples/fs
t_audio = np.arange(total_samples)/fs

L_, R_ = np.meshgrid(np.arange(1,BSound+1), np.arange(1,BSound+1), indexing='ij')
mask = L_ <= R_

frequencies = np.real(MinEnergies[:BSound]*rootFreq)
freq_l = frequencies[L_-1]  # indexing shift
freq_r = frequencies[R_-1]
freq_l = freq_l[mask]
freq_r = freq_r[mask]

freq_r_sample = np.tile(freq_r, (frame_length_samples,1)).T
freq_l_sample = np.tile(freq_l, (frame_length_samples,1)).T

#%% Psi In
PsiIn = (Vectors[:,ind[0]] - Vectors[:,ind[1]])/np.sqrt(2)

#%%  Pure Hamiltonian dynamics
PsiHam, RhoHam = HamiltonianExponentiate(PsiIn,N, tf, Hhat, hbar)

# Define sampling parameters
num_samples = 1000  # Number of samples you want
step_size = max(int(N / num_samples), 1)  # Ensure step_size is at least 1

# Initialize lists and arrays
normPsiHam = []
expectationXHam = []
expectationPHam = []
# Preallocate PositionRepHam with the correct number of columns
PositionRepHam = np.zeros((Nx, num_samples), dtype=complex)

# Iterate over the desired indices
for m, n_ in enumerate(range(0, N, step_size)):
    if m >= num_samples:
        break  # Prevent exceeding the preallocated size

    # Compute norms and expectations
    normPsiHam.append(np.linalg.norm(PsiHam[:, n_]))
    expectationXHam.append(np.real(PsiHam[:, n_].conjugate().T @ Xhat @ PsiHam[:, n_]))
    expectationPHam.append(np.real(PsiHam[:, n_].conjugate().T @ Phat @ PsiHam[:, n_]))

    # Compute position representation
    PositionRepHam[:, m] = HVector @ PsiHam[:, n_]

# Convert lists to numpy arrays
normPsiHam = np.array(normPsiHam)
expectationXHam = np.array(expectationXHam)
expectationPHam = np.array(expectationPHam)
# PositionRepHam is already correctly sized

# Now compute the Wigner function
WigHam = WignerT(PositionRepHam, x, p, hbar)

#%% Hamiltonian Audio

RhoHam = HamiltonianEigenrep(RhoHam, InvVectors, N)
ArgHam = np.angle(RhoHam[:BSound,:BSound,:])
ArgHam = np.unwrap(ArgHam, axis=2)
ArgHam = ArgHam - ArgHam[:,:,0:1]
AbsHam = np.abs(RhoHam[:BSound,:BSound,:])

# Hamiltonian Sound
audio_signal_Ham = np.zeros((total_samples, 2))
max_magnitude_Ham = np.max(AbsHam)
alpha = np.linspace(0,1,frame_length_samples)
up_triangular = np.sum(mask)

for n_ in range(N-1):
    magnitudes0 = AbsHam[:,:,n_][mask]
    phases0 = ArgHam[:,:,n_][mask]
    magnitudes1 = AbsHam[:,:,n_+1][mask]
    phases1 = ArgHam[:,:,n_+1][mask]

    magnitudesSample = np.outer(magnitudes0, (1 - alpha)) + np.outer(magnitudes1, alpha)
    phasesSample = np.outer(phases0, (1 - alpha)) + np.outer(phases1, alpha)

    start_sample = n_*frame_length_samples
    end_sample = start_sample + frame_length_samples

    t_frame = t_audio[start_sample:end_sample]
    t_frame = np.tile(t_frame,(up_triangular,1))

    frame_signalL = magnitudesSample * np.sin(2*np.pi*freq_l_sample*t_frame + phasesSample)
    frame_signalR = magnitudesSample * np.sin(2*np.pi*freq_r_sample*t_frame - phasesSample)

    audio_signal_Ham[start_sample:end_sample,0] = np.sum(frame_signalL, axis=0)
    audio_signal_Ham[start_sample:end_sample,1] = np.sum(frame_signalR, axis=0)

maxAudio = np.max(np.abs(audio_signal_Ham))
if maxAudio > 0:
    audio_signal_Ham = 0.9 * audio_signal_Ham / maxAudio

# Write to a WAV file
sf.write(f'Hamiltonian_EvolutionSound_c2={c2}_c4={c4}.wav', audio_signal_Ham, fs)

#%% Hamiltonian Video

# Setup the figure and axis
fig, ax = plt.subplots(figsize=(10, 8))
cmap = plt.get_cmap('plasma')  # Choose a perceptually uniform colormap

# Initial pcolormesh
W_initial = WigHam[:, :, 0]
c = ax.pcolormesh(x1, p1, W_initial, shading='gouraud', cmap=cmap, vmin=-0.1, vmax=0.3)
fig.colorbar(c, ax=ax)

# Plot the potential contour
xMesh, yMesh = np.meshgrid(np.linspace(-Xview, Xview, Nx), np.linspace(-Xview, Xview, Nx))
Z = 0.5*yMesh**2 + c4 * xMesh**4 - c2 * xMesh**2
contour_levels = 25
ax.contour(xMesh, yMesh, Z, levels=contour_levels, colors='k', linewidths=0.5)

# Initialize expectation value line
expLine, = ax.plot([], [], 'w-', linewidth=2, label='Expectation Values')

# Add a text object for the clock in the top-left corner
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Set plot limits and labels
ax.set_xlim([-Xview, Xview])
ax.set_ylim([-Xview, Xview])
ax.set_xlabel('$x$', fontsize=16)
ax.set_ylabel('$p$', fontsize=16)
title = f'Hamiltonian Wigner Evolution, c₂={c2}, c₄={c4}'
ax.set_title(title, fontsize=20)
ax.legend(loc='upper right')

# Function to initialize the animation
def init():
    c.set_array(WigHam[:, :, 0].ravel())
    expLine.set_data([], [])
    time_text.set_text('T = 0.00')  # Initialize with T = 0.00 or any default value
    return c, expLine, time_text

# Function to update each frame
def animate(i):
    # Update Wigner function
    t = np.real(TSpan[2*i])
    W = WigHam[:, :, i]
    c.set_array(W.ravel())

    # Update expectation value line
    xData = expectationXHam[:i+1]
    yData = expectationPHam[:i+1]
    expLine.set_data(xData, yData)

    # Update the clock text
    time_text.set_text(f't = {t:.2f}')

    return c, expLine, time_text

# Create the animation
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=WigHam.shape[2], interval=1, blit=True)

# Save the animation
video_filename = f'HamiltonianWignerEvolution_c2={c2}_c4={c4}.mp4'
logging.info("Saving video to {video_filename}...")
anim.save(video_filename, writer='ffmpeg', fps=1000/total_duration, dpi=300)
logging.info("Video saved successfully.")

plt.close(fig)  # Close the figure to prevent it from displaying in some environments

# ------------------------------
# Optional: Display the Video in Jupyter (if using Jupyter)
# ------------------------------
# from IPython.display import HTML
# HTML(anim.to_jshtml())

#%% AV combination Ham



video_filename = f'HamiltonianWignerEvolution_c2={c2}_c4={c4}.mp4'
audio_filename = f'Hamiltonian_EvolutionSound_c2={c2}_c4={c4}.wav'
output_filename = f'Hamiltonian_EvolutionAV_c2={c2}_c4={c4}.mp4'

# In this command:
# -c:v copy  will copy the video without re-encoding, preserving video quality.
# -c:a alac  encodes the audio in Apple Lossless (no quality loss).
# -ac 2      ensures the audio is stereo.
#
# Apple Lossless (ALAC) works inside MP4 containers and maintains lossless quality.
# This will re-encode audio from WAV to ALAC, but it remains lossless.
cmd = [
    'ffmpeg',
    '-i', video_filename,
    '-i', audio_filename,
    '-c:v', 'copy',
    '-c:a', 'alac',
    '-ac', '2',
    output_filename
]

# Execute the ffmpeg command
subprocess.run(cmd, check=True)

# Check if the file exists before attempting to delete it
if os.path.exists(video_filename):
    os.remove(video_filename)
    print(f"Deleted {video_filename}")
else:
    print(f"{video_filename} does not exist")
    
    
#%%  Pure SSE dynamics
eta=np.random.normal(0, 1, NSSE)

PsiSSE, RhoSSE = SSEDynamicsSRK(PsiIn, tf/NSSE, tf, HhatGamma,Lhat,hbar, eta)

# Define sampling parameters
num_samples = 1000  # Number of samples you want
step_size = max(int(N / num_samples), 1)  # Ensure step_size is at least 1

# Initialize lists and arrays
normPsiSSE = []
expectationXSSE = []
expectationPSSE = []
# Preallocate PositionRepSSE with the correct number of columns
PositionRepSSE = np.zeros((Nx, num_samples), dtype=complex)

# Iterate over the desired indices
for m, n_ in enumerate(range(0, N, step_size)):
    if m >= num_samples:
        break  # Prevent exceeding the preallocated size

    # Compute norms and expectations
    normPsiSSE.append(np.linalg.norm(PsiSSE[:, n_]))
    expectationXSSE.append(np.real(PsiSSE[:, n_].conjugate().T @ Xhat @ PsiSSE[:, n_]))
    expectationPSSE.append(np.real(PsiSSE[:, n_].conjugate().T @ Phat @ PsiSSE[:, n_]))

    # Compute position representation
    PositionRepSSE[:, m] = HVector @ PsiSSE[:, n_]

# Convert lists to numpy arrays
normPsiSSE = np.array(normPsiSSE)
expectationXSSE = np.array(expectationXSSE)
expectationPSSE = np.array(expectationPSSE)
# PositionRepSSE is already correctly sized

# Now compute the Wigner function
WigSSE = WignerT(PositionRepSSE, x, p, hbar)

#%% SSE Audio

RhoSSE = HamiltonianEigenrep(RhoSSE, InvVectors, N)
ArgSSE = np.angle(RhoSSE[:BSound,:BSound,:])
ArgSSE = np.unwrap(ArgSSE, axis=2)
ArgSSE = ArgSSE - ArgSSE[:,:,0:1]
AbsSSE = np.abs(RhoSSE[:BSound,:BSound,:])

# SSE Sound
audio_signal_SSE = np.zeros((total_samples, 2))
max_magnitude_SSE = np.max(AbsSSE)
alpha = np.linspace(0,1,frame_length_samples)
up_triangular = np.sum(mask)

for n_ in range(N-1):
    magnitudes0 = AbsSSE[:,:,n_][mask]
    phases0 = ArgSSE[:,:,n_][mask]
    magnitudes1 = AbsSSE[:,:,n_+1][mask]
    phases1 = ArgSSE[:,:,n_+1][mask]

    magnitudesSample = np.outer(magnitudes0, (1 - alpha)) + np.outer(magnitudes1, alpha)
    phasesSample = np.outer(phases0, (1 - alpha)) + np.outer(phases1, alpha)

    start_sample = n_*frame_length_samples
    end_sample = start_sample + frame_length_samples

    t_frame = t_audio[start_sample:end_sample]
    t_frame = np.tile(t_frame,(up_triangular,1))

    frame_signalL = magnitudesSample * np.sin(2*np.pi*freq_l_sample*t_frame + phasesSample)
    frame_signalR = magnitudesSample * np.sin(2*np.pi*freq_r_sample*t_frame - phasesSample)

    audio_signal_SSE[start_sample:end_sample,0] = np.sum(frame_signalL, axis=0)
    audio_signal_SSE[start_sample:end_sample,1] = np.sum(frame_signalR, axis=0)

maxAudio = np.max(np.abs(audio_signal_SSE))
if maxAudio > 0:
    audio_signal_SSE = 0.9 * audio_signal_SSE / maxAudio

# Write to a WAV file
sf.write(f'SSE_EvolutionSound_c2={c2}_c4={c4}.wav', audio_signal_SSE, fs)

#%% SSE Video

# Setup the figure and axis
fig, ax = plt.subplots(figsize=(10, 8))
cmap = plt.get_cmap('plasma')  # Choose a perceptually uniform colormap

# Initial pcolormesh
W_initial = WigSSE[:, :, 0]
c = ax.pcolormesh(x1, p1, W_initial, shading='gouraud', cmap=cmap, vmin=-0.1, vmax=0.3)
fig.colorbar(c, ax=ax)

# Plot the potential contour
xMesh, yMesh = np.meshgrid(np.linspace(-Xview, Xview, Nx), np.linspace(-Xview, Xview, Nx))
Z = 0.5*yMesh**2 + c4 * xMesh**4 - c2 * xMesh**2
contour_levels = 25
ax.contour(xMesh, yMesh, Z, levels=contour_levels, colors='k', linewidths=0.5)

# Initialize expectation value line
expLine, = ax.plot([], [], 'w-', linewidth=2, label='Expectation Values')

# Add a text object for the clock in the top-left corner
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Set plot limits and labels
ax.set_xlim([-Xview, Xview])
ax.set_ylim([-Xview, Xview])
ax.set_xlabel('$x$', fontsize=16)
ax.set_ylabel('$p$', fontsize=16)
title = f'SSE Wigner Evolution, c₂={c2}, c₄={c4}'
ax.set_title(title, fontsize=20)
ax.legend(loc='upper right')

# Function to initialize the animation
def init():
    c.set_array(WigSSE[:, :, 0].ravel())
    expLine.set_data([], [])
    time_text.set_text('T = 0.00')  # Initialize with T = 0.00 or any default value
    return c, expLine, time_text

# Function to update each frame
def animate(i):
    # Update Wigner function
    t = np.real(TSpan[2*i])
    W = WigSSE[:, :, i]
    c.set_array(W.ravel())

    # Update expectation value line
    xData = expectationXSSE[:i+1]
    yData = expectationPSSE[:i+1]
    expLine.set_data(xData, yData)

    # Update the clock text
    time_text.set_text(f't = {t:.2f}')

    return c, expLine, time_text

# Create the animation
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=WigSSE.shape[2], interval=1, blit=True)

# Save the animation
video_filename = f'SSEWignerEvolution_c2={c2}_c4={c4}.mp4'
logging.info("Saving video to {video_filename}...")
anim.save(video_filename, writer='ffmpeg', fps=1000/total_duration, dpi=300)
logging.info("Video saved successfully.")

plt.close(fig)  # Close the figure to prevent it from displaying in some environments

# ------------------------------
# Optional: Display the Video in Jupyter (if using Jupyter)
# ------------------------------
# from IPython.display import HTML
# HTML(anim.to_jshtml())

#%% AV combination SSE


video_filename = f'SSEWignerEvolution_c2={c2}_c4={c4}.mp4'
audio_filename = f'SSE_EvolutionSound_c2={c2}_c4={c4}.wav'
output_filename = f'SSE_EvolutionAV_c2={c2}_c4={c4}.mp4'

# In this command:
# -c:v copy  will copy the video without re-encoding, preserving video quality.
# -c:a alac  encodes the audio in Apple Lossless (no quality loss).
# -ac 2      ensures the audio is stereo.
#
# Apple Lossless (ALAC) works inside MP4 containers and maintains lossless quality.
# This will re-encode audio from WAV to ALAC, but it remains lossless.
cmd = [
    'ffmpeg',
    '-i', video_filename,
    '-i', audio_filename,
    '-c:v', 'copy',
    '-c:a', 'alac',
    '-ac', '2',
    output_filename
]

# Execute the ffmpeg command
subprocess.run(cmd, check=True)

# Check if the file exists before attempting to delete it
if os.path.exists(video_filename):
    os.remove(video_filename)
    print(f"Deleted {video_filename}")
else:
    print(f"{video_filename} does not exist")

#%%  Lindblad dynamics

# Compute the Lindblad evolution
PsiInRho=PsiIn.reshape(-1, 1) 
RhoLind = lindblad_exp(PsiInRho @ PsiInRho.conj().T, tf, HhatGamma, Lhat, hbar, N=2000)
print('Lind time below')

# Define sampling parameters
num_samples = 1000  # Number of samples you want
step_size = max(int(N / num_samples), 1)  # Ensure step_size is at least 1

# Initialize lists and arrays to store results
normRhoLind = []
expectationXLind = []
expectationPLind = []

# Preallocate PositionRepLind with the correct dimensions
# Shape: (Nx, Nx, num_samples)
PositionRepLind = np.zeros((Nx, Nx, num_samples), dtype=complex)

# Iterate over the desired indices
for m, n_ in enumerate(range(0, N, step_size)):
    if m >= num_samples:
        break  # Prevent exceeding the preallocated size

    # Extract the current density matrix
    rho_current = RhoLind[:, :, n_]
    rho_current = rho_current/np.trace(rho_current)


    # Compute the trace (norm)
    norm = np.trace(rho_current)
    normRhoLind.append(norm)

    # Compute expectation values
    expectation_X = np.real(np.trace(Xhat @ rho_current))
    expectation_P = np.real(np.trace(Phat @ rho_current))
    expectationXLind.append(expectation_X)
    expectationPLind.append(expectation_P)

    # Compute position representation
    # Assuming HVector is a matrix that transforms rho_current
    PositionRepLind[:, :, m] = HVector @ rho_current @ HVector.conj().T
    PositionRepLind[:, :, m]=PositionRepLind[:, :, m]/np.trace(PositionRepLind[:, :, m])

# Convert lists to numpy arrays for further processing
normRhoLind = np.array(normRhoLind)
expectationXLind = np.array(expectationXLind)
expectationPLind = np.array(expectationPLind)
# PositionRepLind is already a numpy array

# Compute the Wigner function
WigLind = rho_wigner(PositionRepLind, x, p, hbar)

#%% Lind Audio

RhoLind = HamiltonianEigenrep(RhoLind, InvVectors, N)
ArgLind = np.angle(RhoLind[:BSound,:BSound,:])
ArgLind = np.unwrap(ArgLind, axis=2)
ArgLind = ArgLind - ArgLind[:,:,0:1]
AbsLind = np.abs(RhoLind[:BSound,:BSound,:])

# Lind Sound
audio_signal_Lind = np.zeros((total_samples, 2))
max_magnitude_Lind = np.max(AbsLind)
alpha = np.linspace(0,1,frame_length_samples)
up_triangular = np.sum(mask)

for n_ in range(N-1):
    magnitudes0 = AbsLind[:,:,n_][mask]
    phases0 = ArgLind[:,:,n_][mask]
    magnitudes1 = AbsLind[:,:,n_+1][mask]
    phases1 = ArgLind[:,:,n_+1][mask]

    magnitudesSample = np.outer(magnitudes0, (1 - alpha)) + np.outer(magnitudes1, alpha)
    phasesSample = np.outer(phases0, (1 - alpha)) + np.outer(phases1, alpha)

    start_sample = n_*frame_length_samples
    end_sample = start_sample + frame_length_samples

    t_frame = t_audio[start_sample:end_sample]
    t_frame = np.tile(t_frame,(up_triangular,1))

    frame_signalL = magnitudesSample * np.sin(2*np.pi*freq_l_sample*t_frame + phasesSample)
    frame_signalR = magnitudesSample * np.sin(2*np.pi*freq_r_sample*t_frame - phasesSample)

    audio_signal_Lind[start_sample:end_sample,0] = np.sum(frame_signalL, axis=0)
    audio_signal_Lind[start_sample:end_sample,1] = np.sum(frame_signalR, axis=0)

maxAudio = np.max(np.abs(audio_signal_Lind))
if maxAudio > 0:
    audio_signal_Lind = 0.9 * audio_signal_Lind / maxAudio

# Write to a WAV file
sf.write(f'Lind_EvolutionSound_c2={c2}_c4={c4}.wav', audio_signal_Lind, fs)

#%% Lind Video

# Setup the figure and axis
fig, ax = plt.subplots(figsize=(10, 8))
cmap = plt.get_cmap('plasma')  # Choose a perceptually uniform colormap

# Initial pcolormesh
W_initial = WigLind[:, :, 0]
c = ax.pcolormesh(x1, p1, W_initial, shading='gouraud', cmap=cmap, vmin=-0.1, vmax=0.3)
fig.colorbar(c, ax=ax)

# Plot the potential contour
xMesh, yMesh = np.meshgrid(np.linspace(-Xview, Xview, Nx), np.linspace(-Xview, Xview, Nx))
Z = 0.5*yMesh**2 + c4 * xMesh**4 - c2 * xMesh**2
contour_levels = 25
ax.contour(xMesh, yMesh, Z, levels=contour_levels, colors='k', linewidths=0.5)

# Initialize expectation value line
expLine, = ax.plot([], [], 'w-', linewidth=2, label='Expectation Values')

# Add a text object for the clock in the top-left corner
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Set plot limits and labels
ax.set_xlim([-Xview, Xview])
ax.set_ylim([-Xview, Xview])
ax.set_xlabel('$x$', fontsize=16)
ax.set_ylabel('$p$', fontsize=16)
title = f'Lindblad Wigner Evolution, c₂={c2}, c₄={c4}'
ax.set_title(title, fontsize=20)
ax.legend(loc='upper right')

# Function to initialize the animation
def init():
    c.set_array(WigLind[:, :, 0].ravel())
    expLine.set_data([], [])
    time_text.set_text('T = 0.00')  # Initialize with T = 0.00 or any default value
    return c, expLine, time_text

# Function to update each frame
def animate(i):
    # Update Wigner function
    t = np.real(TSpan[2*i])
    W = WigLind[:, :, i]
    c.set_array(W.ravel())

    # Update expectation value line
    xData = expectationXLind[:i+1]
    yData = expectationPLind[:i+1]
    expLine.set_data(xData, yData)

    # Update the clock text
    time_text.set_text(f't = {t:.2f}')

    return c, expLine, time_text

# Create the animation
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=WigLind.shape[2], interval=1, blit=True)

# Save the animation
video_filename = f'LindWignerEvolution_c2={c2}_c4={c4}.mp4'
logging.info("Saving video to {video_filename}...")
anim.save(video_filename, writer='ffmpeg', fps=1000/total_duration, dpi=300)
logging.info("Video saved successfully.")

plt.close(fig)  # Close the figure to prevent it from displaying in some environments

# ------------------------------
# Optional: Display the Video in Jupyter (if using Jupyter)
# ------------------------------
# from IPython.display import HTML
# HTML(anim.to_jshtml())

#%% AV combination Lind


video_filename = f'LindWignerEvolution_c2={c2}_c4={c4}.mp4'
audio_filename = f'Lind_EvolutionSound_c2={c2}_c4={c4}.wav'
output_filename = f'Lind_EvolutionAV_c2={c2}_c4={c4}.mp4'

# In this command:
# -c:v copy  will copy the video without re-encoding, preserving video quality.
# -c:a alac  encodes the audio in Apple Lossless (no quality loss).
# -ac 2      ensures the audio is stereo.
#
# Apple Lossless (ALAC) works inside MP4 containers and maintains lossless quality.
# This will re-encode audio from WAV to ALAC, but it remains lossless.
cmd = [
    'ffmpeg',
    '-i', video_filename,
    '-i', audio_filename,
    '-c:v', 'copy',
    '-c:a', 'alac',
    '-ac', '2',
    output_filename
]

# Execute the ffmpeg command
subprocess.run(cmd, check=True)

# Check if the file exists before attempting to delete it
if os.path.exists(video_filename):
    os.remove(video_filename)
    print(f"Deleted {video_filename}")
else:
    print(f"{video_filename} does not exist")