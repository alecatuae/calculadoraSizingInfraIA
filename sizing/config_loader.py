"""
Carregador de configurações JSON (models, servers, storage).
"""

import json
from pathlib import Path
from typing import Dict

from .models import ModelSpec
from .servers import ServerSpec
from .storage import StorageProfile


class ConfigLoader:
    """Carrega e gerencia especificações de models, servers e storage."""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self._models: Dict[str, ModelSpec] = {}
        self._servers: Dict[str, ServerSpec] = {}
        self._storage: Dict[str, StorageProfile] = {}
    
    def load_models(self, filepath: str = "models.json") -> Dict[str, ModelSpec]:
        """Carrega especificações de modelos do JSON."""
        path = self.base_path / filepath
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Arquivo de modelos não encontrado: {path}\n"
                "Certifique-se de que models.json existe no diretório."
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Erro ao parsear {path}: {e}")
        
        models = {}
        for m in data.get("models", []):
            model = ModelSpec(
                name=m["name"],
                num_layers=m["num_layers"],
                num_key_value_heads=m["num_key_value_heads"],
                head_dim=m["head_dim"],
                max_position_embeddings=m["max_position_embeddings"],
                attention_pattern=m["attention_pattern"],
                hybrid_full_layers=m.get("hybrid_full_layers"),
                hybrid_sliding_layers=m.get("hybrid_sliding_layers"),
                sliding_window=m.get("sliding_window"),
                default_kv_precision=m.get("default_kv_precision", "fp8"),
                total_params_b=m.get("total_params_b"),
                weights_memory_gib_fp16=m.get("weights_memory_gib_fp16"),
                weights_memory_gib_bf16=m.get("weights_memory_gib_bf16"),
                weights_memory_gib_fp8=m.get("weights_memory_gib_fp8"),
                weights_memory_gib_int8=m.get("weights_memory_gib_int8"),
                weights_memory_gib_int4=m.get("weights_memory_gib_int4"),
                default_weights_precision=m.get("default_weights_precision"),
                active_params_b=m.get("active_params_b"),
                notes=m.get("notes", "")
            )
            model.validate()
            models[model.name] = model
        
        self._models = models
        return models
    
    def load_servers(self, filepath: str = "servers.json") -> Dict[str, ServerSpec]:
        """Carrega especificações de servidores do JSON."""
        path = self.base_path / filepath
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Arquivo de servidores não encontrado: {path}\n"
                "Certifique-se de que servers.json existe no diretório."
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Erro ao parsear {path}: {e}")
        
        servers = {}
        for s in data.get("servers", []):
            server = ServerSpec(
                name=s["name"],
                gpus=s["gpus"],
                hbm_per_gpu_gb=s["hbm_per_gpu_gb"],
                power_kw_idle=s.get("power_kw_idle"),
                power_kw_max=s.get("power_kw_max"),
                rack_units_u=s.get("rack_units_u"),
                nvlink_bandwidth_tbps=s.get("nvlink_bandwidth_tbps"),
                system_memory_tb=s.get("system_memory_tb"),
                total_hbm_gb=s.get("total_hbm_gb"),
                notes=s.get("notes", "")
            )
            server.validate()
            servers[server.name] = server
        
        self._servers = servers
        return servers
    
    def load_storage(self, filepath: str = "storage.json") -> Dict[str, StorageProfile]:
        """Carrega perfis de storage do JSON."""
        path = self.base_path / filepath
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Arquivo de storage não encontrado: {path}\n"
                "Certifique-se de que storage.json existe no diretório."
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Erro ao parsear {path}: {e}")
        
        profiles = {}
        for p in data.get("profiles", []):
            # Retrocompatibilidade: converter Gbps para MB/s se necessário
            throughput_read_mbps = p.get("throughput_read_mbps", 0.0)
            throughput_write_mbps = p.get("throughput_write_mbps", 0.0)
            
            # Se não tem MB/s mas tem Gbps (formato antigo), converter
            if throughput_read_mbps == 0.0 and "throughput_read_gbps" in p:
                throughput_read_mbps = p["throughput_read_gbps"] * 125.0  # Gbps → MB/s
            if throughput_write_mbps == 0.0 and "throughput_write_gbps" in p:
                throughput_write_mbps = p["throughput_write_gbps"] * 125.0
            
            profile = StorageProfile(
                name=p["name"],
                type=p["type"],
                capacity_total_tb=p.get("capacity_total_tb", 0.0),
                usable_capacity_tb=p.get("usable_capacity_tb", 0.0),
                iops_read_max=p.get("iops_read_max", 0),
                iops_write_max=p.get("iops_write_max", 0),
                throughput_read_mbps=throughput_read_mbps,
                throughput_write_mbps=throughput_write_mbps,
                block_size_kb_read=p.get("block_size_kb_read", 0.0),
                block_size_kb_write=p.get("block_size_kb_write", 0.0),
                latency_read_ms_p50=p.get("latency_read_ms_p50", 0.0),
                latency_read_ms_p99=p.get("latency_read_ms_p99", 0.0),
                latency_write_ms_p50=p.get("latency_write_ms_p50", 0.0),
                latency_write_ms_p99=p.get("latency_write_ms_p99", 0.0),
                rack_units_u=p.get("rack_units_u", 0),
                power_kw=p.get("power_kw", 0.0),
                notes=p.get("notes", "")
            )
            profile.validate()
            profiles[profile.name] = profile
        
        self._storage = profiles
        return profiles
    
    def get_model(self, name: str) -> ModelSpec:
        """Retorna modelo por nome."""
        if name not in self._models:
            available = ", ".join(self._models.keys())
            raise ValueError(
                f"Modelo '{name}' não encontrado.\n"
                f"Modelos disponíveis: {available}"
            )
        return self._models[name]
    
    def get_server(self, name: str) -> ServerSpec:
        """Retorna servidor por nome."""
        if name not in self._servers:
            available = ", ".join(self._servers.keys())
            raise ValueError(
                f"Servidor '{name}' não encontrado.\n"
                f"Servidores disponíveis: {available}"
            )
        return self._servers[name]
    
    def get_storage(self, name: str) -> StorageProfile:
        """Retorna perfil de storage por nome."""
        if name not in self._storage:
            available = ", ".join(self._storage.keys())
            raise ValueError(
                f"Perfil de storage '{name}' não encontrado.\n"
                f"Perfis disponíveis: {available}"
            )
        return self._storage[name]
