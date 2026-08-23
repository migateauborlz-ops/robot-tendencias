"""
Pruebas de la deteccion de intencion de compra (Modulo 2).

Ejercitan la logica lexica, que es estatica y no requiere cargar ningun modelo:
corren en menos de un segundo y no cuestan nada.

Run:  ./venv/Scripts/python.exe test_intent_detection.py
"""
import sys

from src.nlp_layer import NLPLayer, INTENT_THRESHOLD

PASSED = 0
FAILED = 0


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [OK]   {name}")
    else:
        FAILED += 1
        print(f"  [FALLA] {name}" + (f" -> {detail}" if detail else ""))


cue = NLPLayer._intent_cue_strength


def test_preguntas_explicitas():
    print("\n1. Preguntas transaccionales explicitas (fuerza 1,0)")
    for texto in ["cuanto vale?", "cuánto cuesta ese", "que precio tiene",
                  "a como lo dejas", "donde lo compro", "dónde la venden",
                  "como lo pido", "pasame el link", "how much is it",
                  "where can i buy this", "link please"]:
        check(f'"{texto}"', cue(texto) == 1.0, f"fuerza {cue(texto)}")


def test_senales_debiles():
    print("\n2. Senales de intencion mas debiles (fuerza 0,7)")
    for texto in ["lo necesito ya", "me interesa", "hacen envios",
                  "contra entrega?", "esta disponible", "i need it",
                  "is it in stock", "add to cart"]:
        check(f'"{texto}"', cue(texto) == 0.7, f"fuerza {cue(texto)}")


def test_sin_intencion():
    print("\n3. Comentarios sin intencion comercial (fuerza 0,0)")
    for texto in ["jajaja que video tan gracioso", "this is so funny hahaha",
                  "que porqueria no sirve", "incredible", "so easy to use",
                  "love it", "first!", "quien mas vino por el audio"]:
        check(f'"{texto}"', cue(texto) == 0.0, f"fuerza {cue(texto)}")


def test_limites_de_palabra():
    print("\n4. Limites de palabra (el bug de la busqueda por subcadena)")
    check('"needle" no dispara "need"', cue("she is threading a needle") == 0.0,
          f"fuerza {cue('she is threading a needle')}")
    check('"hayan" no dispara "hay"', cue("ojala hayan visto el partido") == 0.0,
          f"fuerza {cue('ojala hayan visto el partido')}")
    check('"valentina" no dispara "vale"', cue("valentina se ve linda") == 0.0,
          f"fuerza {cue('valentina se ve linda')}")
    check('"buying" si dispara', cue("thinking about buying it") > 0.0)


def test_bilinguismo():
    print("\n5. Cobertura bilingue")
    check("Espanol con tilde", cue("cuánto vale") == 1.0)
    check("Espanol sin tilde", cue("cuanto vale") == 1.0)
    check("Ingles equivalente", cue("how much") == 1.0)
    check("El caso que antes daba cero", cue("cuanto vale? lo necesito ya") == 1.0,
          "era 0.000 con la lista solo en ingles")


def test_composicion_del_puntaje():
    print("\n6. Composicion del puntaje final")
    # score = cue * (0.6 + 0.4 * sentimiento)
    for etiqueta, sent, esperado_min, esperado_max in [
            ("explicita + positivo", 1.0, 0.99, 1.01),
            ("explicita + neutro", 0.5, 0.79, 0.81),
            ("explicita + negativo", 0.0, 0.59, 0.61),
            ("debil + positivo", 1.0, 0.69, 0.71),
            ("debil + negativo", 0.0, 0.41, 0.43)]:
        c = 1.0 if "explicita" in etiqueta else 0.7
        score = c * (0.6 + 0.4 * sent)
        check(f"{etiqueta} -> {score:.2f}", esperado_min <= score <= esperado_max)

    check("Una pregunta explicita siempre alcanza el umbral",
          1.0 * (0.6 + 0.4 * 0.0) >= INTENT_THRESHOLD)
    check("Sin senal lexica el puntaje es cero, sea cual sea el sentimiento",
          cue("this is so funny hahaha") * (0.6 + 0.4 * 1.0) == 0.0)


if __name__ == "__main__":
    print("=" * 68)
    print("Pruebas de deteccion de intencion de compra (sin modelo, sin costo)")
    print("=" * 68)
    test_preguntas_explicitas()
    test_senales_debiles()
    test_sin_intencion()
    test_limites_de_palabra()
    test_bilinguismo()
    test_composicion_del_puntaje()
    print("\n" + "=" * 68)
    print(f"RESULTADO: {PASSED} correctas, {FAILED} fallidas")
    print("=" * 68)
    sys.exit(1 if FAILED else 0)
