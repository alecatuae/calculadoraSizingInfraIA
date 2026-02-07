#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_sizing.py - Script de testes para o sistema de dimensionamento
"""

import subprocess
import json
import sys


def run_sizing_test(name, args, expected_checks=None):
    """
    Executa um teste de sizing e valida resultados.
    
    Args:
        name: Nome do teste
        args: Lista de argumentos para sizing.py
        expected_checks: Função opcional para validar resultados
    """
    print(f"\n{'='*80}")
    print(f"TESTE: {name}")
    print(f"{'='*80}")
    
    cmd = ["python3", "sizing.py"] + args + ["--json-only"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
        
        # Parse JSON
        data = json.loads(output)
        
        # Imprimir resumo
        print(f"✓ Modelo: {data['model']['name']}")
        print(f"✓ Servidor: {data['server']['name']}")
        print(f"✓ Storage: {data['storage']['name']}")
        print(f"✓ Concorrência: {data['parameters']['concurrency']:,}")
        print(f"✓ Contexto Efetivo: {data['parameters']['effective_context']:,}")
        print(f"✓ KV por Sessão: {data['results']['kv_per_session_gib']} GiB")
        print(f"✓ Sessões por Nó: {data['results']['sessions_per_node']}")
        print(f"✓ Nós Finais: {data['results']['nodes_final']}")
        
        if data['warnings']:
            print(f"⚠ Avisos: {len(data['warnings'])}")
            for i, warning in enumerate(data['warnings'][:3], 1):
                print(f"  {i}. {warning[:80]}...")
        
        # Validações customizadas
        if expected_checks:
            expected_checks(data)
        
        print(f"✅ PASSOU")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ FALHOU: {e}")
        print(f"Stderr: {e.stderr}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ FALHOU: Erro ao parsear JSON: {e}")
        print(f"Output: {result.stdout}")
        return False
    except Exception as e:
        print(f"❌ FALHOU: {e}")
        return False


def main():
    """Executa bateria de testes."""
    print("=" * 80)
    print("BATERIA DE TESTES - SISTEMA DE DIMENSIONAMENTO LLM")
    print("=" * 80)
    
    tests_passed = 0
    tests_failed = 0
    
    # ========================================================================
    # TESTE 1: Cenário base - opt-oss-120b + dgx300 + fp8 + N+1
    # ========================================================================
    if run_sizing_test(
        "Cenário Base - 120B + DGX300 + FP8 + N+1",
        [
            "--model", "opt-oss-120b",
            "--server", "dgx300",
            "--storage", "profile_default",
            "--concurrency", "1000",
            "--effective-context", "131072",
            "--kv-precision", "fp8",
            "--ha", "n+1"
        ],
        expected_checks=lambda d: (
            assert_equal(d['results']['nodes_final'], 3, "Nós finais deve ser 3") and
            assert_equal(d['parameters']['kv_precision'], 'fp8', "Precisão deve ser fp8")
        )
    ):
        tests_passed += 1
    else:
        tests_failed += 1
    
    # ========================================================================
    # TESTE 2: Cenário econômico - 20B + dgx200 + fp8
    # ========================================================================
    if run_sizing_test(
        "Cenário Econômico - 20B + DGX200 + FP8",
        [
            "--model", "opt-oss-20b",
            "--server", "dgx200",
            "--storage", "profile_default",
            "--concurrency", "1000",
            "--effective-context", "32768",
            "--kv-precision", "fp8",
            "--ha", "none"
        ],
        expected_checks=lambda d: (
            assert_equal(d['results']['nodes_final'], 1, "Nós finais deve ser 1") and
            assert_greater(d['results']['sessions_per_node'], 1000, "Sessões/nó > 1000")
        )
    ):
        tests_passed += 1
    else:
        tests_failed += 1
    
    # ========================================================================
    # TESTE 3: Alta precisão - FP16 vs FP8
    # ========================================================================
    if run_sizing_test(
        "Alta Precisão - FP16 (dobra memória)",
        [
            "--model", "opt-oss-20b",
            "--server", "dgx200",
            "--storage", "profile_default",
            "--concurrency", "500",
            "--effective-context", "65536",
            "--kv-precision", "fp16",
            "--ha", "none"
        ],
        expected_checks=lambda d: (
            assert_has_warning(d['warnings'], "fp16", "Deve avisar sobre fp16") and
            assert_greater(d['results']['kv_per_session_gib'], 1.0, "KV/sessão > 1.0 GiB")
        )
    ):
        tests_passed += 1
    else:
        tests_failed += 1
    
    # ========================================================================
    # TESTE 4: Context overflow - clamping
    # ========================================================================
    if run_sizing_test(
        "Context Overflow - Clamping",
        [
            "--model", "opt-oss-120b",
            "--server", "dgx300",
            "--storage", "profile_default",
            "--concurrency", "500",
            "--effective-context", "999999",  # Muito maior que max_position_embeddings
            "--kv-precision", "fp8",
            "--ha", "none"
        ],
        expected_checks=lambda d: (
            assert_has_warning(d['warnings'], "excede max_position_embeddings", "Deve avisar sobre overflow")
        )
    ):
        tests_passed += 1
    else:
        tests_failed += 1
    
    # ========================================================================
    # TESTE 5: Storage de rede - alerta de latência
    # ========================================================================
    if run_sizing_test(
        "Storage de Rede - Alertas",
        [
            "--model", "opt-oss-120b",
            "--server", "dgx300",
            "--storage", "profile_network_ssd",
            "--concurrency", "1000",
            "--effective-context", "131072",
            "--kv-precision", "fp8",
            "--ha", "none"
        ],
        expected_checks=lambda d: (
            assert_equal(d['storage']['type'], 'network_ssd', "Storage deve ser network_ssd")
        )
    ):
        tests_passed += 1
    else:
        tests_failed += 1
    
    # ========================================================================
    # TESTE 6: Alta concorrência - múltiplos nós
    # ========================================================================
    if run_sizing_test(
        "Alta Concorrência - Múltiplos Nós",
        [
            "--model", "opt-oss-120b",
            "--server", "dgx300",
            "--storage", "profile_default",
            "--concurrency", "5000",
            "--effective-context", "131072",
            "--kv-precision", "fp8",
            "--peak-headroom-ratio", "0.30",
            "--ha", "n+1"
        ],
        expected_checks=lambda d: (
            assert_greater(d['results']['nodes_final'], 5, "Nós finais > 5")
        )
    ):
        tests_passed += 1
    else:
        tests_failed += 1
    
    # ========================================================================
    # TESTE 7: Contexto pequeno - máxima eficiência
    # ========================================================================
    if run_sizing_test(
        "Contexto Pequeno - Máxima Eficiência",
        [
            "--model", "opt-oss-20b",
            "--server", "dgx200",
            "--storage", "profile_default",
            "--concurrency", "2000",
            "--effective-context", "4096",
            "--kv-precision", "fp8",
            "--ha", "none"
        ],
        expected_checks=lambda d: (
            assert_less(d['results']['kv_per_session_gib'], 0.1, "KV/sessão < 0.1 GiB") and
            assert_greater(d['results']['sessions_per_node'], 2000, "Sessões/nó > 2000")
        )
    ):
        tests_passed += 1
    else:
        tests_failed += 1
    
    # ========================================================================
    # TESTE 8: Cloud storage - validação de perfil
    # ========================================================================
    if run_sizing_test(
        "Cloud Storage - Perfil Premium",
        [
            "--model", "opt-oss-20b",
            "--server", "dgx200",
            "--storage", "profile_cloud_premium",
            "--concurrency", "500",
            "--effective-context", "32768",
            "--kv-precision", "fp8",
            "--ha", "none"
        ],
        expected_checks=lambda d: (
            assert_equal(d['storage']['type'], 'cloud_block_storage', "Storage deve ser cloud")
        )
    ):
        tests_passed += 1
    else:
        tests_failed += 1
    
    # ========================================================================
    # RESUMO
    # ========================================================================
    print("\n" + "=" * 80)
    print("RESUMO DOS TESTES")
    print("=" * 80)
    print(f"✅ Testes Passados: {tests_passed}")
    print(f"❌ Testes Falhos: {tests_failed}")
    print(f"📊 Total: {tests_passed + tests_failed}")
    print(f"Taxa de Sucesso: {100 * tests_passed / (tests_passed + tests_failed):.1f}%")
    print("=" * 80)
    
    return 0 if tests_failed == 0 else 1


# ============================================================================
# FUNÇÕES DE VALIDAÇÃO
# ============================================================================
def assert_equal(actual, expected, message):
    """Valida igualdade."""
    if actual == expected:
        print(f"  ✓ {message}: {actual}")
        return True
    else:
        print(f"  ✗ {message}: esperado {expected}, obtido {actual}")
        return False


def assert_greater(actual, threshold, message):
    """Valida que valor é maior que threshold."""
    if actual > threshold:
        print(f"  ✓ {message}: {actual} > {threshold}")
        return True
    else:
        print(f"  ✗ {message}: {actual} não é > {threshold}")
        return False


def assert_less(actual, threshold, message):
    """Valida que valor é menor que threshold."""
    if actual < threshold:
        print(f"  ✓ {message}: {actual} < {threshold}")
        return True
    else:
        print(f"  ✗ {message}: {actual} não é < {threshold}")
        return False


def assert_has_warning(warnings, keyword, message):
    """Valida que existe warning contendo keyword."""
    for w in warnings:
        if keyword.lower() in w.lower():
            print(f"  ✓ {message}")
            return True
    print(f"  ✗ {message}: keyword '{keyword}' não encontrada em warnings")
    return False


if __name__ == "__main__":
    sys.exit(main())
