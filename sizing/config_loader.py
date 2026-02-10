"""
Carregador de configurações JSON (models, servers, storage) com validação de schema.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

from .models import ModelSpec
from .servers import ServerSpec
from .storage import StorageProfile
from .validator import validate_models, validate_servers, validate_storage_profiles


class ConfigLoader:
    """Carrega e gerencia especificações de models, servers e storage com validação."""
    
    def __init__(self, base_path: str = ".", validate: bool = True):
        """
        Args:
            base_path: Caminho base para os arquivos JSON
            validate: Se True, valida schemas ao carregar
        """
        self.base_path = Path(base_path)
        self.validate = validate
        
        # Cache
        self._models: Dict[str, ModelSpec] = {}
        self._servers: Dict[str, ServerSpec] = {}
        self._storage: Dict[str, StorageProfile] = {}
        
        # Dados brutos para validação
        self._models_data: List[Dict[str, Any]] = []
        self._servers_data: List[Dict[str, Any]] = []
        self._storage_data: List[Dict[str, Any]] = []
    
    def load_models(self, filepath: str = "models.json") -> Dict[str, ModelSpec]:
        """Carrega especificações de modelos do JSON."""
        path = self.base_path / filepath
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"❌ Arquivo de modelos não encontrado: {path}\n"
                "Certifique-se de que models.json existe no diretório."
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"❌ Erro ao parsear {path}: {e}")
        
        # Armazenar dados brutos
        self._models_data = data.get("models", [])
        
        # Validar schema se solicitado
        if self.validate:
            errors, warnings = validate_models(self._models_data)
            if errors:
                error_msg = "\n".join(errors)
                raise ValueError(f"❌ Erros de validação em models.json:\n{error_msg}")
        
        # Parsear modelos
        models = {}
        for m in self._models_data:
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
                active_params_b=m.get("active_params_b"),
                weights_memory_gib_fp16=m.get("weights_memory_gib_fp16"),
                weights_memory_gib_bf16=m.get("weights_memory_gib_bf16"),
                weights_memory_gib_fp8=m.get("weights_memory_gib_fp8"),
                weights_memory_gib_int8=m.get("weights_memory_gib_int8"),
                weights_memory_gib_int4=m.get("weights_memory_gib_int4"),
                default_weights_precision=m.get("default_weights_precision", "fp8"),
                notes=m.get("notes", "")
            )
            model.validate()
            models[model.name.lower()] = model
        
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
                f"❌ Arquivo de servidores não encontrado: {path}\n"
                "Certifique-se de que servers.json existe no diretório."
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"❌ Erro ao parsear {path}: {e}")
        
        # Armazenar dados brutos
        self._servers_data = data.get("servers", [])
        
        # Validar schema se solicitado
        if self.validate:
            errors, warnings = validate_servers(self._servers_data)
            if errors:
                error_msg = "\n".join(errors)
                raise ValueError(f"❌ Erros de validação em servers.json:\n{error_msg}")
        
        # Parsear servidores
        servers = {}
        for s in self._servers_data:
            server = ServerSpec(
                name=s["name"],
                gpus=s["gpus"],
                hbm_per_gpu_gb=s["hbm_per_gpu_gb"],
                rack_units_u=s.get("rack_units_u", 10),
                power_kw_max=s["power_kw_max"],
                heat_output_btu_hr_max=s.get("heat_output_btu_hr_max"),
                notes=s.get("notes", "")
            )
            server.validate()
            servers[server.name.lower()] = server
        
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
                f"❌ Arquivo de storage não encontrado: {path}\n"
                "Certifique-se de que storage.json existe no diretório."
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"❌ Erro ao parsear {path}: {e}")
        
        # Armazenar dados brutos
        self._storage_data = data.get("profiles", [])
        
        # Validar schema se solicitado
        if self.validate:
            errors, warnings = validate_storage_profiles(self._storage_data)
            if errors:
                error_msg = "\n".join(errors)
                raise ValueError(f"❌ Erros de validação em storage.json:\n{error_msg}")
        
        # Parsear perfis de storage
        profiles = {}
        for p in self._storage_data:
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
            profiles[profile.name.lower()] = profile
        
        self._storage = profiles
        return profiles
    
    def get_model(self, name: str) -> ModelSpec:
        """Busca modelo por nome (case-insensitive)."""
        if not self._models:
            self.load_models()
        
        name_normalized = name.lower()
        if name_normalized not in self._models:
            available = ", ".join(self._models.keys())
            raise ValueError(
                f"❌ Modelo '{name}' não encontrado em models.json.\n"
                f"Modelos disponíveis: {available}"
            )
        return self._models[name_normalized]
    
    def get_server(self, name: str) -> ServerSpec:
        """Busca servidor por nome (case-insensitive)."""
        if not self._servers:
            self.load_servers()
        
        name_normalized = name.lower()
        if name_normalized not in self._servers:
            available = ", ".join(self._servers.keys())
            raise ValueError(
                f"❌ Servidor '{name}' não encontrado em servers.json.\n"
                f"Servidores disponíveis: {available}"
            )
        return self._servers[name_normalized]
    
    def get_storage(self, name: str) -> StorageProfile:
        """Busca perfil de storage por nome (case-insensitive)."""
        if not self._storage:
            self.load_storage()
        
        name_normalized = name.lower()
        if name_normalized not in self._storage:
            available = ", ".join(self._storage.keys())
            raise ValueError(
                f"❌ Perfil de storage '{name}' não encontrado em storage.json.\n"
                f"Perfis disponíveis: {available}"
            )
        return self._storage[name_normalized]
    
    def get_raw_data(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Retorna dados brutos (não parseados) para validação.
        
        Returns:
            (models_data, servers_data, storage_data)
        """
        return self._models_data, self._servers_data, self._storage_data
