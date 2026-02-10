"""
Validação de schemas e constraints para models.json, servers.json e storage.json.
"""

from typing import Dict, Any, List, Tuple
from .schemas import MODEL_SCHEMA, SERVER_SCHEMA, STORAGE_SCHEMA


class ValidationError(Exception):
    """Erro de validação de schema."""
    pass


def validate_object(
    obj: Dict[str, Any],
    schema: Dict[str, Any],
    obj_type: str,
    obj_name: str = "unknown"
) -> List[str]:
    """
    Valida um objeto contra um schema.
    
    Args:
        obj: Objeto a validar (dict do JSON)
        schema: Schema de referência
        obj_type: Tipo do objeto ("model", "server", "storage")
        obj_name: Nome do objeto (para mensagens de erro)
    
    Returns:
        Lista de erros (vazia se válido)
    """
    errors = []
    
    # 1. Validar campos obrigatórios
    for field, expected_type in schema["required"].items():
        if field not in obj:
            errors.append(
                f"[{obj_type}:{obj_name}] Campo obrigatório ausente: '{field}'. "
                f"Atualize o JSON ou forneça override via CLI."
            )
        else:
            # Validar tipo
            value = obj[field]
            if not _check_type(value, expected_type):
                errors.append(
                    f"[{obj_type}:{obj_name}] Campo '{field}' tem tipo inválido. "
                    f"Esperado: {_type_to_str(expected_type)}, Recebido: {type(value).__name__}"
                )
    
    # 2. Validar campos opcionais (se presentes)
    for field, expected_type in schema.get("optional", {}).items():
        if field in obj:
            value = obj[field]
            if not _check_type(value, expected_type):
                errors.append(
                    f"[{obj_type}:{obj_name}] Campo opcional '{field}' tem tipo inválido. "
                    f"Esperado: {_type_to_str(expected_type)}, Recebido: {type(value).__name__}"
                )
    
    # 3. Validar enums
    for field, valid_values in schema.get("enums", {}).items():
        if field in obj:
            value = obj[field]
            # Normalizar para comparação case-insensitive
            if isinstance(value, str):
                value_normalized = value.lower()
                valid_normalized = [v.lower() for v in valid_values]
                if value_normalized not in valid_normalized:
                    errors.append(
                        f"[{obj_type}:{obj_name}] Campo '{field}' tem valor inválido: '{value}'. "
                        f"Valores aceitos: {', '.join(valid_values)}"
                    )
    
    # 4. Validar constraints
    for constraint in schema.get("constraints", []):
        try:
            if not constraint["check"](obj):
                errors.append(
                    f"[{obj_type}:{obj_name}] Constraint '{constraint['name']}' falhou: "
                    f"{constraint['error']}"
                )
        except Exception as e:
            errors.append(
                f"[{obj_type}:{obj_name}] Erro ao validar constraint '{constraint['name']}': {str(e)}"
            )
    
    return errors


def _check_type(value: Any, expected_type: Any) -> bool:
    """
    Verifica se value tem o tipo esperado.
    
    Suporta:
    - Tipos simples: str, int, float, bool
    - Tuplas de tipos: (int, float) significa "int OU float"
    - type(None) para aceitar None
    """
    if isinstance(expected_type, tuple):
        # expected_type é uma tupla de tipos alternativos
        return any(_check_type(value, t) for t in expected_type)
    else:
        return isinstance(value, expected_type)


def _type_to_str(expected_type: Any) -> str:
    """Converte tipo esperado para string legível."""
    if isinstance(expected_type, tuple):
        parts = []
        for t in expected_type:
            if t is type(None):
                parts.append("null")
            else:
                parts.append(t.__name__)
        return " | ".join(parts)
    else:
        if expected_type is type(None):
            return "null"
        return expected_type.__name__


def validate_models(models: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """
    Valida lista de modelos.
    
    Returns:
        (erros, warnings)
    """
    errors = []
    warnings = []
    
    # Verificar nomes duplicados
    names = [m.get("name", "").lower() for m in models]
    duplicates = [name for name in names if names.count(name) > 1]
    if duplicates:
        errors.append(
            f"[models.json] Nomes duplicados encontrados: {', '.join(set(duplicates))}. "
            "Cada modelo deve ter um nome único."
        )
    
    # Validar cada modelo
    for model in models:
        model_name = model.get("name", "unknown")
        model_errors = validate_object(model, MODEL_SCHEMA, "model", model_name)
        errors.extend(model_errors)
    
    return errors, warnings


def validate_servers(servers: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """
    Valida lista de servidores.
    
    Returns:
        (erros, warnings)
    """
    errors = []
    warnings = []
    
    # Verificar nomes duplicados
    names = [s.get("name", "").lower() for s in servers]
    duplicates = [name for name in names if names.count(name) > 1]
    if duplicates:
        errors.append(
            f"[servers.json] Nomes duplicados encontrados: {', '.join(set(duplicates))}. "
            "Cada servidor deve ter um nome único."
        )
    
    # Validar cada servidor
    for server in servers:
        server_name = server.get("name", "unknown")
        server_errors = validate_object(server, SERVER_SCHEMA, "server", server_name)
        errors.extend(server_errors)
    
    return errors, warnings


def validate_storage_profiles(profiles: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """
    Valida lista de perfis de storage.
    
    Returns:
        (erros, warnings)
    """
    errors = []
    warnings = []
    
    # Verificar nomes duplicados
    names = [p.get("name", "").lower() for p in profiles]
    duplicates = [name for name in names if names.count(name) > 1]
    if duplicates:
        errors.append(
            f"[storage.json] Nomes duplicados encontrados: {', '.join(set(duplicates))}. "
            "Cada perfil deve ter um nome único."
        )
    
    # Validar cada perfil
    for profile in profiles:
        profile_name = profile.get("name", "unknown")
        profile_errors = validate_object(profile, STORAGE_SCHEMA, "storage", profile_name)
        errors.extend(profile_errors)
    
    return errors, warnings


def validate_all_configs(
    models: List[Dict[str, Any]],
    servers: List[Dict[str, Any]],
    storage_profiles: List[Dict[str, Any]]
) -> Tuple[List[str], List[str]]:
    """
    Valida todos os arquivos de configuração.
    
    Returns:
        (erros, warnings)
    """
    all_errors = []
    all_warnings = []
    
    # Validar models
    model_errors, model_warnings = validate_models(models)
    all_errors.extend(model_errors)
    all_warnings.extend(model_warnings)
    
    # Validar servers
    server_errors, server_warnings = validate_servers(servers)
    all_errors.extend(server_errors)
    all_warnings.extend(server_warnings)
    
    # Validar storage
    storage_errors, storage_warnings = validate_storage_profiles(storage_profiles)
    all_errors.extend(storage_errors)
    all_warnings.extend(storage_warnings)
    
    return all_errors, all_warnings


def print_validation_report(errors: List[str], warnings: List[str]) -> bool:
    """
    Imprime relatório de validação.
    
    Returns:
        True se validação passou (sem erros), False caso contrário
    """
    print("\n" + "=" * 100)
    print("VALIDAÇÃO DE SCHEMAS E CONSTRAINTS")
    print("=" * 100)
    
    if warnings:
        print(f"\n⚠️  {len(warnings)} WARNING(S):")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")
    
    if errors:
        print(f"\n❌ {len(errors)} ERRO(S) ENCONTRADO(S):")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        print("\n" + "=" * 100)
        print("❌ VALIDAÇÃO FALHOU")
        print("=" * 100 + "\n")
        return False
    else:
        if not warnings:
            print("\n✅ Todos os arquivos de configuração são válidos.")
        else:
            print("\n✅ Validação passou (com warnings).")
        print("=" * 100 + "\n")
        return True
