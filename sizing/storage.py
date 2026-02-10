"""
Definições de perfis de storage e suas especificações de I/O.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class StorageProfile:
    """Especificação de um perfil de storage para operação de inferência."""
    
    name: str
    type: str  # ex: "nvme_local", "network_ssd", "cloud_block_storage"
    
    # Capacidade (TB)
    capacity_total_tb: float = 0.0
    usable_capacity_tb: float = 0.0
    
    # IOPS máximos
    iops_read_max: int = 0
    iops_write_max: int = 0
    
    # Throughput (MB/s) - base decimal, para cálculo com block size
    throughput_read_mbps: float = 0.0
    throughput_write_mbps: float = 0.0
    
    # Block Size (KB) - para validação de consistência física
    block_size_kb_read: float = 0.0
    block_size_kb_write: float = 0.0
    
    # Latency percentiles (ms)
    latency_read_ms_p50: float = 0.0
    latency_read_ms_p99: float = 0.0
    latency_write_ms_p50: float = 0.0
    latency_write_ms_p99: float = 0.0
    
    # Físico (datacenter)
    rack_units_u: int = 0
    power_kw: float = 0.0
    
    # Metadata
    notes: str = ""
    
    def validate(self) -> None:
        """Valida especificação do perfil de storage."""
        if self.capacity_total_tb < 0:
            raise ValueError(f"Storage {self.name}: capacity_total_tb must be >= 0")
        if self.usable_capacity_tb < 0:
            raise ValueError(f"Storage {self.name}: usable_capacity_tb must be >= 0")
        if self.usable_capacity_tb > self.capacity_total_tb:
            raise ValueError(f"Storage {self.name}: usable_capacity_tb cannot exceed capacity_total_tb")
        if self.iops_read_max < 0:
            raise ValueError(f"Storage {self.name}: iops_read_max must be >= 0")
        if self.iops_write_max < 0:
            raise ValueError(f"Storage {self.name}: iops_write_max must be >= 0")
        if self.throughput_read_mbps < 0:
            raise ValueError(f"Storage {self.name}: throughput_read_mbps must be >= 0")
        if self.throughput_write_mbps < 0:
            raise ValueError(f"Storage {self.name}: throughput_write_mbps must be >= 0")
