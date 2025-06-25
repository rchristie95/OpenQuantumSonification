globals().clear()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import eig, expm, kron
import soundfile as sf
from numpy.fft import fft, ifft, fftshift, ifftshift
import logging
import subprocess
import os


# For animations, one could use matplotlib.animation, but here we will just store frames
# from matplotlib.animation import FFMpegWriter


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


def SSEDynamicsSRK_MultiL(PsiIn: np.ndarray, dt: float, tf: float, 
                          Hhat: np.ndarray, L_ops: list, 
                          hbar: float, eta: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Simulate the Stochastic Schrödinger Equation (SSE) dynamics with multiple Lindblad operators
    using a Stochastic Runge-Kutta (SRK) scheme.

    Parameters:
    - PsiIn (np.ndarray): Initial state vector (shape: (bSize,))
    - dt (float): Time step
    - tf (float): Final time
    - Hhat (np.ndarray): Hamiltonian matrix (bSize x bSize)
    - L_ops (list of np.ndarray): List of Lindblad operators, each (bSize x bSize)
    - hbar (float): Reduced Planck's constant
    - eta (np.ndarray): Noise array of shape (N, M), where M = len(L_ops)
      and N = number of steps. eta[n,j] is the noise at step n for Lindblad operator j.

    Returns:
    - PsiOut (np.ndarray): Time-evolved state vectors (shape: (bSize, num_saved_steps))
    - RhoOut (np.ndarray): Time-evolved density matrices (shape: (bSize, bSize, num_saved_steps))
    """
    # Total number of time steps
    N = int(round(tf / dt))
    bSize = len(PsiIn)
    M = len(L_ops)  # number of Lindblad operators

    Psi0 = PsiIn.copy()
    Sqdt = np.sqrt(dt)

    # We'll store at most 2000 steps (as in your original code)
    num_steps = 2000
    PsiOut = np.zeros((bSize, num_steps), dtype=complex)
    RhoOut = np.zeros((bSize, bSize, num_steps), dtype=complex)
    PsiOut[:, 0] = PsiIn
    RhoOut[:, :, 0] = np.outer(PsiIn, PsiIn.conjugate())

    # Determine how often results are stored.  Keeping ``modfactor`` as an
    # integer avoids floating point comparisons when using the modulus operator.
    modfactor = max(int(N // 2000), 1)
    m = 1

    for n in range(2, N + 1):
        # Extract noise increments for this step for each Lindblad operator
        eta_step = eta[n - 1, :]  # shape (M,)
        
        # Compute the drift and stochastic terms at Psi0
        DPsi0 = DriftMulti(Psi0, Hhat, L_ops, hbar)
        # For the SRK scheme, we need the stochastic increments
        SPsi0_list = [StochMultiSingle(Psi0, L, hbar) for L in L_ops]

        # Compute etaSq_j for each Lindblad operator
        # etaSq_j = 0.5*(eta_j^2 - 1)
        etaSq = 0.5 * (eta_step**2 - 1.0)  # shape (M,)

        # Build Psi1
        Psi1 = Psi0 + DPsi0 * dt

        # For Psi2 and Psi3, we must consider each operator separately and combine:
        # Psi2 = Psi0 + DPsi0*dt + sum_j SPsi0_j * Sqdt * etaSq_j
        # Psi3 = Psi0 + DPsi0*dt - sum_j SPsi0_j * Sqdt * etaSq_j
        Psi2 = Psi0 + DPsi0 * dt
        Psi3 = Psi0 + DPsi0 * dt
        for j in range(M):
            Psi2 += SPsi0_list[j] * Sqdt * etaSq[j]
            Psi3 += -SPsi0_list[j] * Sqdt * etaSq[j]

        # Now calculate T1, T2, T3
        # T1 involves Drift at Psi1
        DPsi1 = DriftMulti(Psi1, Hhat, L_ops, hbar)
        T1 = Psi0 + 0.5 * (DPsi0 + DPsi1) * dt

        # T2 and T3 combine stochastic increments
        # T2 = sum_j SPsi0_j * eta_j * Sqdt
        T2 = np.zeros_like(Psi0, dtype=complex)
        for j in range(M):
            T2 += SPsi0_list[j] * eta_step[j] * Sqdt

        # T3: 0.5 * Sqdt * sum_j (Stoch(Psi2, L_j)-Stoch(Psi3, L_j))
        T3 = np.zeros_like(Psi0, dtype=complex)
        for j, L in enumerate(L_ops):
            SPsi2_j = StochMultiSingle(Psi2, L, hbar)
            SPsi3_j = StochMultiSingle(Psi3, L, hbar)
            T3 += 0.5 * Sqdt * (SPsi2_j - SPsi3_j)

        # Update the state
        Psi0 = T1 + T2 + T3
        Psi0 = Psi0 / np.linalg.norm(Psi0)

        # Save state
        if n % modfactor == 0:
            if m < num_steps:
                PsiOut[:, m] = Psi0
                RhoOut[:, :, m] = np.outer(Psi0, Psi0.conjugate())
                m += 1
            else:
                break

    # Trim arrays
    PsiOut = PsiOut[:, :m]
    RhoOut = RhoOut[:, :, :m]

    return PsiOut, RhoOut


def DriftMulti(Psi: np.ndarray, H: np.ndarray, L_ops: list, hbar: float) -> np.ndarray:
    """
    Compute the deterministic drift component with multiple Lindblad operators.

    Drift = [(-iH) - 1/2 Σ_j L_j^\dagger L_j + Σ_j (EPsi_j^* L_j) - 1/2 Σ_j |EPsi_j|^2 I] * Psi / hbar

    where EPsi_j = <Psi|L_j|Psi>
    """
    bSize = H.shape[0]
    EPsi_list = []
    for L in L_ops:
        EPsi_list.append(Psi.conjugate().T @ (L @ Psi))
    
    # Compute sum of L_j^\dagger L_j
    sum_LdagL = np.zeros((bSize, bSize), dtype=complex)
    # Compute sum of EPsi_j^* L_j
    sum_EPsiconj_L = np.zeros((bSize, bSize), dtype=complex)
    # Compute sum of |EPsi_j|^2
    sum_absEPsi2 = 0.0

    for j, L in enumerate(L_ops):
        EPsi_j = EPsi_list[j]
        sum_LdagL += L.conjugate().T @ L
        sum_EPsiconj_L += EPsi_j.conjugate() * L
        sum_absEPsi2 += np.abs(EPsi_j)**2

    # Construct Drift operator
    DriftOp = (-1j * H - 0.5 * sum_LdagL + sum_EPsiconj_L - 0.5 * sum_absEPsi2 * np.eye(bSize))
    return (DriftOp @ Psi) / hbar


def StochMultiSingle(Psi: np.ndarray, L: np.ndarray, hbar: float) -> np.ndarray:
    """
    Compute the stochastic increment for a single Lindblad operator L.

    Stoch(L) = (L - EPsi I) Psi * sqrt(1/hbar), where EPsi = <Psi|L|Psi>
    """
    EPsi = Psi.conjugate().T @ (L @ Psi)
    bSize = len(Psi)
    return (L - EPsi * np.eye(bSize)) @ Psi * np.sqrt(1/hbar)


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
## Assuming all your previous imports and functions (HamiltonianExponentiate, lindblad_exp, etc.)
# remain the same. We'll redefine lindblad_exp to handle multiple Lindblad operators.

# Pauli matrices
sx = np.array([[0, 1],
               [1, 0]], dtype=complex)

sy = np.array([[0, -1j],
               [1j, 0]], dtype=complex)

sz = np.array([[1, 0],
               [0, -1]], dtype=complex)

id2 = np.eye(2, dtype=complex)

def kron_n(op_list):
    out = op_list[0]
    for op in op_list[1:]:
        out = np.kron(out, op)
    return out

def embed_operator(op, L, site):
    # Places 'op' at a specific site in an L-site spin chain
    ops = []
    for i in range(L):
        if i == site:
            ops.append(op)
        else:
            ops.append(id2)
    return kron_n(ops)

def build_spin_chain_hamiltonian_xx(L, J=1.0):
    """
    Build an XX model Hamiltonian:
    H = sum_{j} (J/2)*(sigma_j^x sigma_{j+1}^x + sigma_j^y sigma_{j+1}^y)
    """
    dim = 2**L
    H = np.zeros((dim, dim), dtype=complex)
    for j in range(L-1):
        Sx_j = embed_operator(sx, L, j)
        Sx_j1 = embed_operator(sx, L, j+1)
        Sy_j = embed_operator(sy, L, j)
        Sy_j1 = embed_operator(sy, L, j+1)
        # Add XX coupling
        H += 0.5*J*(Sx_j @ Sx_j1 + Sy_j @ Sy_j1)
    return H

def build_lindblad_operators_xx(L, Gamma=0.1):
    """
    Example Lindblad operators:
    L1 = sqrt(Gamma)*sigma^+ at site 0 (inject spin-up)
    L2 = sqrt(Gamma)*sigma^- at site L-1 (remove spin-down)
    """
    splus = np.array([[0, 1],
                      [0, 0]], dtype=complex)
    sminus = np.array([[0, 0],
                       [1, 0]], dtype=complex)

    L_ops = []
    L_ops.append(np.sqrt(Gamma)*embed_operator(splus, L, 0))
    L_ops.append(np.sqrt(Gamma)*embed_operator(sminus, L, L-1))
    return L_ops

def build_spin_chain_hamiltonian_xxz(L, J=1.0, Delta=np.cos(np.pi/4)):
    """
    Build an XXZ Heisenberg Hamiltonian:
    H = sum_{j} J*(sigma_j^x sigma_{j+1}^x + sigma_j^y sigma_{j+1}^y + Delta*(sigma_j^z sigma_{j+1}^z - I))
    
    Parameters:
    - L (int): Number of spins in the chain
    - J (float): Exchange interaction strength
    - Delta (float): Anisotropy parameter (Delta = cos(eta))
    
    Returns:
    - H (numpy.ndarray): The Hamiltonian matrix of size 2^L x 2^L
    """
    dim = 2**L
    H = np.zeros((dim, dim), dtype=complex)
    
    # Define Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    for j in range(L-1):
        # Operators on site j
        Sx_j = embed_operator(sx, L, j)
        Sy_j = embed_operator(sy, L, j)
        Sz_j = embed_operator(sz, L, j)
        
        # Operators on site j+1
        Sx_j1 = embed_operator(sx, L, j+1)
        Sy_j1 = embed_operator(sy, L, j+1)
        Sz_j1 = embed_operator(sz, L, j+1)
        
        # XX + YY coupling
        H += J * (Sx_j @ Sx_j1 + Sy_j @ Sy_j1)
        
        # ZZ coupling with anisotropy Delta and constant shift
        H += J * Delta * (Sz_j @ Sz_j1 - embed_operator(I, L, j) @ embed_operator(I, L, j+1))
        #H += J * Delta * (Sz_j @ Sz_j1 - embed_operator(I, L, j) @ embed_operator(I, L, j+1))

    return H

def build_lindblad_operators_shs(L, alpha_L, beta_L, alpha_R, beta_R, r, Phi, Gamma=0.1):
    """
    Build Lindblad operators D_L and D_R to stabilize the Spin Helix State (SHS).
    
    Parameters:
    - L (int): Number of spins in the chain
    - alpha_L, beta_L (float): Coupling constants for the left boundary
    - alpha_R, beta_R (float): Coupling constants for the right boundary
    - r (float): Parameter related to the SHS properties
    - Phi (float): Twist angle defining the helix structure
    - Gamma (float): Overall scaling factor for the Lindblad operators
    
    Returns:
    - L_ops (list of numpy.ndarray): List containing D_L and D_R Lindblad operators
    """
    # Define Pauli matrices and raising/lowering operators
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    splus = np.array([[0, 1], [0, 0]], dtype=complex)
    sminus = np.array([[0, 0], [1, 0]], dtype=complex)
    
    # Identity operator for embedding
    def embed_op(operator, L, site):
        """Embed a single-site operator into the full chain Hilbert space."""
        op = 1
        for i in range(L):
            if i == site:
                op = np.kron(op, operator)
            else:
                op = np.kron(op, I)
        return op
    
    L_ops = []
    
    # Construct D_L
    # D_L = alpha_L * (r * v1^- * sigma1^+) - beta_L * (n1^- - r * sigma1^-)
    # Assuming v1^- and n1^- are specific operators; since they are not defined in the paper excerpt,
    # we'll interpret v1^- as a lowering operator and n1^- as number operator minus identity
    # This interpretation may vary based on the paper's definitions
    
    # For the purpose of this implementation, let's assume:
    # v_j^- = sminus
    # n_j^- = (sz_j - I)/2, which counts the number of spin-downs
    
    v1_minus = embed_op(sminus, L, 0)
    n1_minus = 0.5 * (embed_op(sz, L, 0) - embed_op(np.eye(2), L, 0))
    
    sigma1_plus = embed_op(splus, L, 0)
    sigma1_minus = embed_op(sminus, L, 0)
    
    D_L = alpha_L * (r * v1_minus @ sigma1_plus) - beta_L * (n1_minus - r * sigma1_minus)
    D_L = np.sqrt(Gamma) * D_L  # Scale by sqrt(Gamma)
    L_ops.append(D_L)
    
    # Construct D_R
    # D_R = alpha_R * (r * vN^- * e^{-iPhi} * sigmaN^+) - beta_R * (nN^- - r * e^{iPhi} * sigmaN^-)
    
    # Operators on the last site (site L-1)
    vN_minus = embed_op(sminus, L, L-1)
    nN_minus = 0.5 * (embed_op(sz, L, L-1) - embed_op(np.eye(2), L, L-1))
    
    sigmaN_plus = embed_op(splus, L, L-1)
    sigmaN_minus = embed_op(sminus, L, L-1)
    
    # Phase factors
    phase = np.exp(-1j * Phi)
    phase_conj = np.exp(1j * Phi)
    
    D_R = alpha_R * (r * vN_minus * phase * sigmaN_plus) - beta_R * (nN_minus - r * phase_conj * sigmaN_minus)
    D_R = np.sqrt(Gamma) * D_R  # Scale by sqrt(Gamma)
    L_ops.append(D_R)
    
    return L_ops



def lindblad_exp_multiple(RhoIn, tf, Hhat, L_ops, hbar, N=2000):
    """
    Modified version of lindblad_exp to handle multiple Lindblad operators L_ops (list).
    dρ/dt = -i[H, ρ] + Σ_j (L_j ρ L_j^† - 0.5 {L_j^† L_j, ρ})

    Parameters:
    - RhoIn: Initial density matrix (bSize x bSize)
    - tf: final time
    - Hhat: Hamiltonian (bSize x bSize)
    - L_ops: list of Lindblad operators
    - hbar: Planck constant
    - N: number of time steps

    Returns:
    - Rho: evolved density matrices (bSize x bSize x N)
    """
    bSize = Hhat.shape[0]
    TSpan = np.linspace(0, tf, N)
    dt = TSpan[1] - TSpan[0]
    Id = np.eye(bSize, dtype=complex)

    # Construct the Liouvillian superoperator M
    # Vectorization: vec(A) => M vec(rho)
    # M = -i/hbar (I⊗H - H^T⊗I) + Σ_j [ (1/hbar)* (conj(L_j)⊗L_j) 
    #    - (1/(2hbar)) (I⊗L_j^†L_j + (L_j^†L_j)^T⊗I ) ]

    # Summation over all L_ops
    sum_conjL_L = 0
    sum_LdagL = 0
    for Lj in L_ops:
        sum_conjL_L += kron(np.conj(Lj), Lj)
        LdagL = (Lj.conjugate().T @ Lj)
        sum_LdagL += LdagL

    M = (-1j/hbar)*(kron(Id, Hhat) - kron(Hhat.T, Id)) \
        + (1/hbar)*sum_conjL_L \
        - (1/(2*hbar))*(kron(Id, sum_LdagL) + kron(sum_LdagL.T, Id))

    # Initial conditions
    Rho = np.zeros((bSize, bSize, N), dtype=complex)
    Rho[:, :, 0] = RhoIn
    Vt = np.zeros((bSize*bSize, N), dtype=complex)
    Vt[:, 0] = RhoIn.flatten(order='F')

    # Propagator
    UProp = expm(M*dt)

    for n in range(1, N):
        Vt[:, n] = UProp @ Vt[:, n-1]
        Rho_n = Vt[:, n].reshape((bSize, bSize), order='F')
        # Ensure Hermiticity and normalization
        Rho_n = 0.5*(Rho_n + Rho_n.conjugate().T)
        Rho_n = Rho_n / np.trace(Rho_n)
        Rho[:, :, n] = Rho_n

        if n % 500 == 0:
            print(f"{n}/{N} steps completed")

    return Rho

# ------------------------------
# Main execution: Spin Helix
# ------------------------------

L = 4      # chain length
sites=range(L)
J = 1.0    # coupling strength
hbar = 1.0
N = 2000
Delta = np.cos(np.pi / 4)

# Lindblad parameters
alpha_L = 1.0
beta_L = 1.0
alpha_R = 1.0
beta_R = 1.0
r = 0.5
Phi = np.pi / 2  # 90 degrees twist
Gamma = 0.2

# Build Hamiltonian
Hhat = build_spin_chain_hamiltonian_xxz(L, J=J, Delta=Delta)

# Build Lindblad operators
Lhat_ops = build_lindblad_operators_shs(L, alpha_L, beta_L, alpha_R, beta_R, r, Phi, Gamma=Gamma)

NSSE=40000

# Hhat = build_spin_chain_hamiltonian(L, J=J)
# Lhat_ops = build_lindblad_operators(L, Gamma=Gamma)

# Convert eigenvalues and eigenvectors to real numbers if possible
# This step is optional and depends on whether Hhat is truly Hermitian
logging.basicConfig(level=logging.INFO)
Energies, Vectors = np.linalg.eig(Hhat)

# Sorting eigenvalues in ascending order and rearranging eigenvectors
sorted_indices = np.argsort(Energies)  # Get the indices that would sort the eigenvalues
Energies =np.real( Energies[sorted_indices])  # Sorted eigenvalues
Vectors = Vectors[:, sorted_indices]  # Corresponding rearranged eigenvectors
Energies=Energies-Energies[0]+1
deltaE = Energies[1]-Energies[0]
t_tunnel = np.real(hbar*np.pi/deltaE)
# tf=8*t_tunnel
tf=200

TSpan = np.linspace(0, tf, N)
num_frames = 1000  # Number of frames you want

# Normalize eigenvectors (optional but recommended for numerical stability)
Vectors = Vectors / np.linalg.norm(Vectors, axis=0)

# Verify the sorting
logging.info("First 10 sorted eigenvalues:")
logging.info(Energies[:10])

InvVectors = np.linalg.inv(Vectors)

# PsiIn = (Vectors[:,0]+Vectors[:,1]+Vectors[:,2]+Vectors[:,3] )/np.sqrt(4)
PsiIn = np.sum(Vectors, axis=1) / np.sqrt(2**L)

# Initial state: maximally mixed
PsiInRho=PsiIn.reshape(-1, 1) 
RhoIn=PsiInRho @ PsiInRho.conj().T

# Thermal density matrix construction at β = 1
RhoIn = expm(-Hhat / 10)           # Compute matrix exponential: e^(-H/β), here β = 1
RhoIn = RhoIn / np.trace(RhoIn)   # Normalize to ensure Tr(RhoIn) = 1


#%% Sonification of Arrays
BSound = 2**L
rootFreq = 60  # in Hz
fs = 44100
duration_per_frame = 1/rootFreq
frame_length_samples = round(duration_per_frame*fs)
total_samples = N*frame_length_samples
total_duration = total_samples/fs
t_audio = np.arange(total_samples)/fs

L_, R_ = np.meshgrid(np.arange(1,BSound+1), np.arange(1,BSound+1), indexing='ij')
mask = L_ <= R_

frequencies = np.real(Energies[:BSound]*rootFreq)
freq_l = frequencies[L_-1]  # indexing shift
freq_r = frequencies[R_-1]
freq_l = freq_l[mask]
freq_r = freq_r[mask]

freq_r_sample = np.tile(freq_r, (frame_length_samples,1)).T
freq_l_sample = np.tile(freq_l, (frame_length_samples,1)).T

#%%  Pure Hamiltonian dynamics

PsiHam, RhoHam = HamiltonianExponentiate(PsiIn,N, tf, Hhat, hbar)
video_filename = 'Hamiltonian_MagnetizationEvolution.mp4'
audio_filename = 'Hamiltonian_MagnetizationEvolutionSound.wav'
output_filename = 'Hamiltonian_MagnetizationAV.mp4'

# Check if the file exists before attempting to delete it
if os.path.exists(video_filename):
    os.remove(video_filename)
    os.remove(audio_filename)
    print(f"Deleted {video_filename} and {audio_filename}  ")
else:
    print(f"{video_filename} does not exist")

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
sf.write(audio_filename, audio_signal_Ham, fs)

#%% Hamiltonian Video


fig, ax = plt.subplots(figsize=(8, 6))

# Initialize lines for Mx, My, and Mz
line_mz, = ax.plot([], [], 'o-', label='Mz')
line_mx, = ax.plot([], [], 'o-', label='Mx')
line_my, = ax.plot([], [], 'o-', label='My')

ax.set_xlabel('Site', fontsize=14)
ax.set_ylabel('Magnetization', fontsize=14)
ax.set_title('Hamiltonian Magnetization Evolution', fontsize=16)
ax.set_xlim(1, L)
ax.set_xticks(range(1,L+1))  # Sets ticks at positions [0, 1, ..., L-1]

# Adjust the y-limits as needed based on your expected magnetization values
ax.set_ylim(-1.1, 1.1)  
ax.legend(loc='upper right')

# Optional: Add a time text
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

def init():
    # Initialize empty data for the lines
    line_mz.set_data([], [])
    line_mx.set_data([], [])
    line_my.set_data([], [])
    time_text.set_text('')
    return line_mz, line_mx, line_my, time_text

def animate(i):
    # Extract the density matrix at time step i
    rho = RhoHam[:, :,2*i]
    t = np.real(TSpan[2*i])

    # Compute magnetization at each site
    mz_vals = []
    mx_vals = []
    my_vals = []
    for site in range(L):
        Sz_site = embed_operator(sz, L, site)
        Sx_site = embed_operator(sx, L, site)
        Sy_site = embed_operator(sy, L, site)
        
        mz_vals.append(np.real(np.trace(rho @ Sz_site)))
        mx_vals.append(np.real(np.trace(rho @ Sx_site)))
        my_vals.append(np.real(np.trace(rho @ Sy_site)))
    
    # Update the line data
    line_mz.set_data(range(1,L+1), mz_vals)
    line_mx.set_data(range(1,L+1), mx_vals)
    line_my.set_data(range(1,L+1), my_vals)
    
    # Update the clock text
    time_text.set_text(f't = {t:.2f}')
    
    return line_mz, line_mx, line_my, time_text

# Create the animation
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=num_frames, interval=1, blit=True)

# Save the animation to a file
print(f"Saving video to {video_filename}...")
anim.save(video_filename, writer='ffmpeg', fps=1000/total_duration, dpi=200)
print("Video saved successfully.")

plt.close(fig)  # Close the figure if running in a non-interactive environment

# If you are in a Jupyter notebook, you could display it with:
# from IPython.display import HTML
# HTML(anim.to_jshtml())

#%% AV combination Ham

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
eta = np.random.normal(0, 1, (NSSE, 2))  # Shape: (2, NSSE)

PsiSSE, RhoSSE = SSEDynamicsSRK_MultiL(PsiIn, tf/NSSE, tf, Hhat,Lhat_ops,hbar, eta)
video_filename = 'SSE_MagnetizationEvolution.mp4'
audio_filename = 'SSE_MagnetizationEvolutionSound.wav'
output_filename = 'SSE_MagnetizationAV.mp4'

# Check if the file exists before attempting to delete it
if os.path.exists(video_filename):
    os.remove(video_filename)
    os.remove(audio_filename)
    print(f"Deleted {video_filename} and {audio_filename}  ")
else:
    print(f"{video_filename} does not exist")
    
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
sf.write(audio_filename, audio_signal_SSE, fs)

#%% SSE Video


fig, ax = plt.subplots(figsize=(8, 6))

# Initialize lines for Mx, My, and Mz
line_mz, = ax.plot([], [], 'o-', label='Mz')
line_mx, = ax.plot([], [], 'o-', label='Mx')
line_my, = ax.plot([], [], 'o-', label='My')

ax.set_xlabel('Site', fontsize=14)
ax.set_ylabel('Magnetization', fontsize=14)
ax.set_title('SSE Magnetization Evolution', fontsize=16)
ax.set_xlim(1, L)
ax.set_xticks(range(1,L+1))  # Sets ticks at positions [0, 1, ..., L-1]

# Adjust the y-limits as needed based on your expected magnetization values
ax.set_ylim(-1.1, 1.1)  
ax.legend(loc='upper right')

# Optional: Add a time text
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

def init():
    # Initialize empty data for the lines
    line_mz.set_data([], [])
    line_mx.set_data([], [])
    line_my.set_data([], [])
    time_text.set_text('')
    return line_mz, line_mx, line_my, time_text

def animate(i):
    # Extract the density matrix at time step i
    rho = RhoSSE[:, :,2*i]
    t = np.real(TSpan[2*i])

    # Compute magnetization at each site
    mz_vals = []
    mx_vals = []
    my_vals = []
    for site in range(L):
        Sz_site = embed_operator(sz, L, site)
        Sx_site = embed_operator(sx, L, site)
        Sy_site = embed_operator(sy, L, site)
        
        mz_vals.append(np.real(np.trace(rho @ Sz_site)))
        mx_vals.append(np.real(np.trace(rho @ Sx_site)))
        my_vals.append(np.real(np.trace(rho @ Sy_site)))
    
    # Update the line data
    line_mz.set_data(range(1,L+1), mz_vals)
    line_mx.set_data(range(1,L+1), mx_vals)
    line_my.set_data(range(1,L+1), my_vals)
    
    # Update the clock text
    time_text.set_text(f't = {t:.2f}')
    
    return line_mz, line_mx, line_my, time_text

# Create the animation
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=num_frames, interval=1, blit=True)

# Save the animation to a file
print(f"Saving video to {video_filename}...")
anim.save(video_filename, writer='ffmpeg', fps=1000/total_duration, dpi=200)
print("Video saved successfully.")

plt.close(fig)  # Close the figure if running in a non-interactive environment

# If you are in a Jupyter notebook, you could display it with:
# from IPython.display import HTML
# HTML(anim.to_jshtml())

#%% AV combination SSE

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
RhoLind  = lindblad_exp_multiple(RhoIn, tf, Hhat, Lhat_ops, hbar, N=N)

print('Lind time below')

video_filename = 'Lind_MagnetizationEvolution.mp4'
audio_filename = 'Lind_MagnetizationEvolutionSound.wav'
output_filename = 'Lind_MagnetizationAV.mp4'

# Check if the file exists before attempting to delete it
if os.path.exists(video_filename):
    os.remove(video_filename)
    os.remove(audio_filename)
    print(f"Deleted {video_filename} and {audio_filename}  ")
else:
    print(f"{video_filename} does not exist")
    
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
sf.write(audio_filename, audio_signal_Lind, fs)

#%% Lind Video


fig, ax = plt.subplots(figsize=(8, 6))

# Initialize lines for Mx, My, and Mz
line_mz, = ax.plot([], [], 'o-', label='Mz')
line_mx, = ax.plot([], [], 'o-', label='Mx')
line_my, = ax.plot([], [], 'o-', label='My')

ax.set_xlabel('Site', fontsize=14)
ax.set_ylabel('Magnetization', fontsize=14)
ax.set_title('Lindblad Magnetization Evolution', fontsize=16)
ax.set_xlim(1, L)
ax.set_xticks(range(1,L+1))  # Sets ticks at positions [0, 1, ..., L-1]

# Adjust the y-limits as needed based on your expected magnetization values
ax.set_ylim(-1.1, 1.1)  
ax.legend(loc='upper right')

# Optional: Add a time text
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

def init():
    # Initialize empty data for the lines
    line_mz.set_data([], [])
    line_mx.set_data([], [])
    line_my.set_data([], [])
    time_text.set_text('')
    return line_mz, line_mx, line_my, time_text

def animate(i):
    # Extract the density matrix at time step i
    rho = RhoLind[:, :,2*i]
    t = np.real(TSpan[2*i])

    # Compute magnetization at each site
    mz_vals = []
    mx_vals = []
    my_vals = []
    for site in range(L):
        Sz_site = embed_operator(sz, L, site)
        Sx_site = embed_operator(sx, L, site)
        Sy_site = embed_operator(sy, L, site)
        
        mz_vals.append(np.real(np.trace(rho @ Sz_site)))
        mx_vals.append(np.real(np.trace(rho @ Sx_site)))
        my_vals.append(np.real(np.trace(rho @ Sy_site)))
    
    # Update the line data
    line_mz.set_data(range(1,L+1), mz_vals)
    line_mx.set_data(range(1,L+1), mx_vals)
    line_my.set_data(range(1,L+1), my_vals)
    
    # Update the clock text
    time_text.set_text(f't = {t:.2f}')
    
    return line_mz, line_mx, line_my, time_text

# Create the animation
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=num_frames, interval=1, blit=True)

# Save the animation to a file
print(f"Saving video to {video_filename}...")
anim.save(video_filename, writer='ffmpeg', fps=1000/total_duration, dpi=200)
print("Video saved successfully.")

plt.close(fig)  # Close the figure if running in a non-interactive environment

# If you are in a Jupyter notebook, you could display it with:
# from IPython.display import HTML
# HTML(anim.to_jshtml())

#%% AV combination Lind

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