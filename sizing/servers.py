"""
Definições de servidores GPU e suas especificações de hardware.
"""

from dataclasses import dataclass
from typing import Optional


# Constante de conversão
GB_TO_GIB = 10**9 / 2**30  # 0.931322574615478515625


@dataclass
class ServerSpec:
    """Especificação de hardware de um servidor GPU."""
    
    name: str
    gpus: int
    hbm_per_gpu_gb: float
    
    # Physical specs
    power_kw_idle: Optional[float] = None
    power_kw_max: Optional[float] = None
    rack_units_u: Optional[int] = None
    heat_output_btu_hr_max: Optional[float] = None
    
    # Optional networking/memory
    nvlink_bandwidth_tbps: Optional[float] = None
    system_memory_tb: Optional[float] = None
    total_hbm_gb: Optional[float] = None
    
    # Metadata
    notes: str = ""
    
    def __post_init__(self):
        """Calcula total_hbm_gb se não fornecido."""
        if self.total_hbm_gb is None:
            self.total_hbm_gb = self.gpus * self.hbm_per_gpu_gb
    
    @property
    def total_hbm_gib(self) -> float:
        """Retorna HBM total em GiB."""
        return self.total_hbm_gb * GB_TO_GIB
    
    def validate(self) -> None:
        """Valida especificação do servidor."""
        if self.gpus <= 0:
            raise ValueError(f"Server {self.name}: gpus must be > 0")
        if self.hbm_per_gpu_gb <= 0:
            raise ValueError(f"Server {self.name}: hbm_per_gpu_gb must be > 0")
        
        # Validar se fornecido
        if self.power_kw_max is not None and self.power_kw_max <= 0:
            raise ValueError(f"Server {self.name}: power_kw_max must be > 0")
        if self.rack_units_u is not None and self.rack_units_u <= 0:
            raise ValueError(f"Server {self.name}: rack_units_u must be > 0")
