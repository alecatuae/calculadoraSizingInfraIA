"""
Definições de modelos LLM e suas especificações arquiteturais.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelSpec:
    """Especificação arquitetural de um modelo LLM."""
    
    name: str
    num_layers: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    attention_pattern: str  # "full", "sliding", "hybrid"
    
    # Hybrid attention
    hybrid_full_layers: Optional[int] = None
    hybrid_sliding_layers: Optional[int] = None
    sliding_window: Optional[int] = None
    
    # KV cache precision
    default_kv_precision: str = "fp8"
    
    # Model weights memory
    total_params_b: Optional[float] = None
    weights_memory_gib_fp16: Optional[float] = None
    weights_memory_gib_bf16: Optional[float] = None
    weights_memory_gib_fp8: Optional[float] = None
    weights_memory_gib_int8: Optional[float] = None
    weights_memory_gib_int4: Optional[float] = None
    default_weights_precision: Optional[str] = None
    
    # Optional metadata
    active_params_b: Optional[float] = None  # For MoE models
    notes: str = ""
    
    def validate(self) -> None:
        """Valida especificação do modelo."""
        if self.attention_pattern == "hybrid":
            if self.hybrid_full_layers is None or self.hybrid_sliding_layers is None:
                raise ValueError(
                    f"Model {self.name}: attention_pattern='hybrid' requires "
                    "hybrid_full_layers and hybrid_sliding_layers"
                )
            if self.hybrid_full_layers + self.hybrid_sliding_layers != self.num_layers:
                raise ValueError(
                    f"Model {self.name}: hybrid_full_layers ({self.hybrid_full_layers}) + "
                    f"hybrid_sliding_layers ({self.hybrid_sliding_layers}) != "
                    f"num_layers ({self.num_layers})"
                )
        
        if self.attention_pattern == "sliding" and self.sliding_window is None:
            raise ValueError(
                f"Model {self.name}: attention_pattern='sliding' requires sliding_window"
            )
    
    def get_weights_memory(self, precision: str) -> Optional[float]:
        """Retorna memória dos pesos em GiB para a precisão especificada."""
        field_map = {
            "fp16": self.weights_memory_gib_fp16,
            "bf16": self.weights_memory_gib_bf16,
            "fp8": self.weights_memory_gib_fp8,
            "int8": self.weights_memory_gib_int8,
            "int4": self.weights_memory_gib_int4,
        }
        return field_map.get(precision)
    
    @staticmethod
    def kv_bytes_per_elem(precision: str) -> int:
        """Retorna bytes por elemento de KV cache para a precisão especificada."""
        precision_map = {
            "fp16": 2,
            "bf16": 2,
            "fp8": 1,
            "int8": 1,
        }
        return precision_map.get(precision, 1)
    
    @staticmethod
    def weights_bytes_per_param(precision: str) -> float:
        """Retorna bytes por parâmetro para a precisão especificada."""
        precision_map = {
            "fp16": 2.0,
            "bf16": 2.0,
            "fp8": 1.0,
            "int8": 1.0,
            "int4": 0.5,
        }
        return precision_map.get(precision, 1.0)
