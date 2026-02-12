"""
Definições de servidores GPU e suas especificações de hardware (estrutura hierárquica).
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any


# Constante de conversão
GB_TO_GIB = 10**9 / 2**30  # 0.931322574615478515625


@dataclass
class CPUSpec:
    """Especificação de CPU."""
    model: Optional[str] = None
    cores_total: Optional[int] = None
    threads_total: Optional[int] = None
    base_frequency_ghz: Optional[float] = None
    max_boost_frequency_ghz: Optional[float] = None


@dataclass
class SystemMemorySpec:
    """Especificação de memória do sistema."""
    capacity_total_tb: Optional[float] = None
    type: Optional[str] = None
    speed_mhz: Optional[int] = None


@dataclass
class GPUSpec:
    """Especificação de GPU."""
    count: int
    model: str
    hbm_per_gpu_gb: float
    total_hbm_gb: Optional[float] = None
    nvlink_bandwidth_tbps_total: Optional[float] = None
    nvlink_generation: Optional[str] = None
    
    def validate(self, server_name: str) -> None:
        """Valida especificação de GPU."""
        if self.count <= 0:
            raise ValueError(f"Server {server_name}: gpu.count must be > 0")
        if self.hbm_per_gpu_gb <= 0:
            raise ValueError(f"Server {server_name}: gpu.hbm_per_gpu_gb must be > 0")
        
        # Validar consistência de total_hbm_gb se existir
        if self.total_hbm_gb is not None:
            expected_total = self.count * self.hbm_per_gpu_gb
            divergence_pct = abs(self.total_hbm_gb - expected_total) / expected_total
            
            if divergence_pct > 0.01:  # Tolerância de 1%
                # Warning e correção automática
                print(
                    f"⚠️  Warning: Server {server_name}: gpu.total_hbm_gb ({self.total_hbm_gb:.1f} GB) "
                    f"diverge do valor derivado ({expected_total:.1f} GB) em {divergence_pct*100:.1f}%. "
                    f"Usando valor derivado como correto."
                )
                self.total_hbm_gb = expected_total
        else:
            # Derivar automaticamente
            self.total_hbm_gb = self.count * self.hbm_per_gpu_gb
    
    @property
    def total_hbm_gib(self) -> float:
        """HBM total em GiB."""
        return self.total_hbm_gb * GB_TO_GIB


@dataclass
class PowerSupplySpec:
    """Especificação de fontes de alimentação."""
    count: Optional[int] = None
    rating_each_watts: Optional[int] = None
    redundancy: Optional[str] = None


@dataclass
class MaxCurrentSpec:
    """Especificação de corrente máxima."""
    _208v_3phase_amps: Optional[int] = None
    _480v_3phase_amps: Optional[int] = None


@dataclass
class PowerSpec:
    """Especificação de consumo elétrico."""
    power_kw_max: float
    power_supplies: Optional[PowerSupplySpec] = None
    input_voltage: Optional[List[str]] = None
    max_current: Optional[MaxCurrentSpec] = None
    
    def validate(self, server_name: str) -> None:
        """Valida especificação de power."""
        if self.power_kw_max <= 0:
            raise ValueError(f"Server {server_name}: power.power_kw_max must be > 0")


@dataclass
class ThermalSpec:
    """Especificação térmica."""
    heat_output_btu_hr_max: Optional[float] = None
    ambient_temp_operating_c_min: Optional[int] = None
    ambient_temp_operating_c_max: Optional[int] = None


@dataclass
class CoolingSpec:
    """Especificação de refrigeração."""
    airflow_cfm: Optional[float] = None
    cooling_type: Optional[str] = None


@dataclass
class StorageSpec:
    """Especificação de storage interno."""
    boot_drives: Optional[str] = None
    internal_nvme_slots: Optional[int] = None
    max_internal_storage_tb: Optional[float] = None


@dataclass
class NetworkingSpec:
    """Especificação de rede."""
    infiniband: Optional[str] = None
    management: Optional[str] = None


@dataclass
class SoftwareSpec:
    """Especificação de software."""
    os_supported: Optional[List[str]] = None
    nvidia_ai_enterprise: Optional[str] = None
    cuda_version: Optional[str] = None


@dataclass
class DimensionsSpec:
    """Dimensões físicas."""
    width: Optional[int] = None
    depth: Optional[int] = None
    height: Optional[int] = None


@dataclass
class PhysicalSpec:
    """Especificação física."""
    dimensions_mm: Optional[DimensionsSpec] = None
    weight_kg_max: Optional[float] = None


@dataclass
class ServerSpec:
    """Especificação completa de hardware de um servidor GPU (estrutura hierárquica)."""
    
    # Identificação
    name: str
    manufacturer: Optional[str] = None
    form_factor: Optional[str] = None
    rack_units_u: int = 10
    
    # Componentes (nested)
    cpu: Optional[CPUSpec] = None
    system_memory: Optional[SystemMemorySpec] = None
    gpu: Optional[GPUSpec] = None
    power: Optional[PowerSpec] = None
    thermal: Optional[ThermalSpec] = None
    cooling: Optional[CoolingSpec] = None
    storage: Optional[StorageSpec] = None
    networking: Optional[NetworkingSpec] = None
    software: Optional[SoftwareSpec] = None
    physical: Optional[PhysicalSpec] = None
    
    # Metadata
    notes: str = ""
    source: Optional[List[str]] = None
    
    def validate(self) -> None:
        """Valida especificação completa do servidor."""
        # Validar campos obrigatórios para cálculos
        if self.gpu is None:
            raise ValueError(f"Server {self.name}: 'gpu' section is required")
        
        if self.power is None:
            raise ValueError(f"Server {self.name}: 'power' section is required")
        
        if self.rack_units_u <= 0:
            raise ValueError(f"Server {self.name}: rack_units_u must be > 0")
        
        # Validar sub-especificações
        self.gpu.validate(self.name)
        self.power.validate(self.name)
    
    @property
    def total_hbm_gib(self) -> float:
        """HBM total em GiB (compatibilidade)."""
        if self.gpu:
            return self.gpu.total_hbm_gib
        return 0.0
    
    @property
    def heat_output_btu_hr_max(self) -> Optional[float]:
        """Heat output máximo (compatibilidade)."""
        if self.thermal:
            return self.thermal.heat_output_btu_hr_max
        return None
