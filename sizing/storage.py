"""
Definições de perfis de storage e suas especificações de I/O.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class StorageProfile:
    """Especificação de um perfil de storage."""
    
    name: str
    type: str  # ex: "nvme", "ssd", "hdd", "network"
    
    # IOPS
    iops_read: Optional[int] = None
    iops_write: Optional[int] = None
    
    # Throughput
    throughput_read_gbps: Optional[float] = None
    throughput_write_gbps: Optional[float] = None
    
    # Latency percentiles (ms)
    latency_read_ms_p50: Optional[float] = None
    latency_read_ms_p99: Optional[float] = None
    latency_write_ms_p50: Optional[float] = None
    latency_write_ms_p99: Optional[float] = None
    
    # Metadata
    notes: str = ""
    
    def validate(self) -> None:
        """Valida especificação do perfil de storage."""
        # Storage é opcional para sizing de KV (que fica em HBM)
        # Validações básicas apenas se valores forem fornecidos
        if self.iops_read is not None and self.iops_read < 0:
            raise ValueError(f"Storage {self.name}: iops_read must be >= 0")
        if self.iops_write is not None and self.iops_write < 0:
            raise ValueError(f"Storage {self.name}: iops_write must be >= 0")
