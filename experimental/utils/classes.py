import torch
from torch.fft import fftfreq

class grid:
    """
    Represents a computational grid with methods to initialize frequency components
    and Laplacian operators.
    """

    def __init__(self, nx=1, ny=1, nz=1, dx=1, dy=1, dz=1):  # constructor
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy
        self.dz = dz

        self.T1 = self.nx * self.dx
        self.T2 = self.ny * self.dy
        self.T3 = self.nz * self.dz

        self.ntot = self.nx * self.ny * self.nz

    def initFREQ(self, choice):
        """
        Initialize frequency components for the grid based on the chosen method.

        Args:
            choice: Method for frequency initialization ('classical' or 'modified').

        Returns:
            Tensor representing frequency components.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if choice == "classical":
            DF1 = 2.0 * torch.pi / self.T1
            DF2 = 2.0 * torch.pi / self.T2
            DF3 = 2.0 * torch.pi / self.T3

            freq = torch.zeros((self.nx, self.ny, self.nz, 3), device=device)

            k = DF1 * fftfreq(self.nx, 1.0 / self.nx).to(device)
            for i in range(self.nx):
                freq[i, :, :, 0] = k[i]

            k = DF2 * fftfreq(self.ny, 1.0 / self.ny).to(device)
            for i in range(self.ny):
                freq[:, i, :, 1] = k[i]

            k = DF3 * fftfreq(self.nz, 1.0 / self.nz).to(device)
            for i in range(self.nz):
                freq[:, :, i, 2] = k[i]

        elif choice == "modified":
            ii = torch.pi * fftfreq(self.nx, 1.0 / self.nx).to(device) / self.nx
            jj = torch.pi * fftfreq(self.ny, 1.0 / self.ny).to(device) / self.ny
            kk = torch.pi * fftfreq(self.nz, 1.0 / self.nz).to(device) / self.nz

            jj, ii, kk = torch.meshgrid(jj, ii, kk, indexing="ij")

            freq = torch.zeros((self.nx, self.ny, self.nz, 3), device=device)
            freq[:, :, :, 0] = (
                2.0 / self.dx * torch.sin(ii) * torch.cos(jj) * torch.cos(kk)
            )
            freq[:, :, :, 1] = (
                2.0 / self.dy * torch.cos(ii) * torch.sin(jj) * torch.cos(kk)
            )
            freq[:, :, :, 2] = (
                2.0 / self.dz * torch.cos(ii) * torch.cos(jj) * torch.sin(kk)
            )

        return freq

    def initFREQ_laplacian(self):
        """
        Initialize the Laplacian operator in the frequency domain.

        Returns:
            Tensor representing the Laplacian operator.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ii = 2.0 * torch.pi * fftfreq(self.nx, 1.0 / self.nx).to(device) / self.nx
        jj = 2.0 * torch.pi * fftfreq(self.ny, 1.0 / self.ny).to(device) / self.ny
        kk = 2.0 * torch.pi * fftfreq(self.nz, 1.0 / self.nz).to(device) / self.nz

        jj, ii, kk = torch.meshgrid(jj, ii, kk, indexing="ij")

        freqLaplacian = (
            2.0 * (torch.cos(ii) - 1.0) / self.dx**2
            + 2.0 * (torch.cos(jj) - 1.0) / self.dy**2
            + 2.0 * (torch.cos(kk) - 1.0) / self.dz**2
        )

        return freqLaplacian


class microstructure:
    """
    Represents the microstructure of a material.
    """

    def __init__(
        self,
        Ifn,
        L,
        label_solid=1,
        label_fluid=0,
        label_B=None,
        micro_permeability=None,
        local_axes=None,
        micro_beta=None,
    ):
        self.Ifn = Ifn  # characteristic function
        self.L = L
        self.label_solid = label_solid
        self.label_fluid = label_fluid
        if label_B != None:
            self.label_B = label_B
        self.micro_permeability = micro_permeability
        self.local_axes = local_axes
        self.micro_beta = micro_beta

        self.nx, self.ny, self.nz = np.shape(Ifn)
        self.dx = L[0] / self.nx
        self.dy = L[1] / self.ny
        self.dz = L[2] / self.nz

    def fft_charact_fct(self):  # fft of the characteristic function
        """
        Compute the FFT of the characteristic function.

        Returns:
            Tensor representing the FFT of the characteristic function.
        """
        return fftn(self.Ifn)

    def vol_frac_solid(self):
        """
        Calculate the volume fraction of the solid phase in the microstructure.

        Returns:
            Float representing the volume fraction of the solid phase.
        """
        return (
            torch.count_nonzero(self.Ifn == self.label_solid).float() / self.Ifn.numel()
        )


class load_fluid_condition:
    """
    Represents the fluid loading conditions, including macro-scale loads and viscosities.
    """

    def __init__(self, macro_load=[1.0, 0.0, 0.0], viscosity=1.0, viscosity_solid=None):
        self.macro_load = macro_load
        self.viscosity = viscosity
        self.viscosity_solid = viscosity_solid


class param_algo:
    """
    Represents the parameters for the numerical algorithm, including convergence criteria
    and acceleration settings.
    """

    def __init__(
        self,
        cv_criterion=1e-4,
        reference_mu0=0.5,
        reference_phi0=None,
        reference_beta0=None,
        reference_ks=0.0,
        velocity_scale=1.0,
        itMax=1000,
        cv_acc=False,
        AA_depth=4,
        AA_increment=4,
    ):
        self.cv_criterion = cv_criterion
        self.reference_mu0 = reference_mu0
        self.itMax = itMax
        self.reference_phi0 = reference_phi0
        self.reference_beta0 = reference_beta0
        self.reference_ks = reference_ks
        self.velocity_scale = velocity_scale
        self.cv_acc = cv_acc
        if cv_acc == True:
            self.AA_depth = AA_depth
            self.AA_increment = AA_increment
