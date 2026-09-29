# Robot de tendencias

Sistema automatizado de inteligencia comercial que detecta tempranamente
tendencias de producto en TikTok e Instagram y las valida contra el volumen de
búsqueda digital, pensado para apoyar la selección de portafolio en PYMES
colombianas de comercio electrónico.

Proyecto aplicado de la Maestría en Analítica de Datos de la Universidad
Central. Daniela Trujillo Villamarín, Miguel Ángel Reina y Gabriel Zambrano
Triviño.

Este repositorio contiene el prototipo que sustenta los capítulos 7 y 8 del
documento de grado. Las cifras que allí se reportan salen de ejecutar el código
que está aquí.

## Qué hace

El sistema encadena cuatro módulos. Cada uno deja su resultado en una base
SQLite, de modo que una falla en un módulo no obliga a repetir los anteriores.

1. **Ingesta.** Recupera publicaciones y comentarios a través de un
   intermediario de extracción, aplica una ventana de recencia de 30 días y
   vincula cada comentario con su publicación de origen.
2. **Procesamiento de lenguaje natural.** Detecta el idioma de cada unidad
   textual y la enruta al modelo correspondiente, extrae entidades candidatas
   de producto y determina intención de compra mediante marcadores léxicos
   explícitos, con el sentimiento como factor modulador.
3. **Validación externa.** Contrasta cada producto candidato contra el índice
   de volumen de búsqueda, mide el rezago entre el mercado de detección y el
   mercado local sobre primeras diferencias, y descarta las correlaciones que
   no superan un umbral de significancia obtenido por simulación.
4. **Puntaje y entrega.** Calcula el Opportunity Score y lo publica en un
   tablero de solo lectura.

El Opportunity Score pondera viralidad 0,40, intención de compra 0,40 y
crecimiento de búsqueda 0,20. Los pesos viven en un único lugar,
`SCORE_WEIGHTS` en `src/database.py`, y el tablero los lee de allí para
descomponer el puntaje.

## Instalación

Requiere Python 3.9 o superior.

```bash
python -m venv venv
venv/Scripts/activate          # en Linux o macOS: source venv/bin/activate
pip install -r requirements.txt
python -m spacy download es_core_news_sm
python -m spacy download en_core_web_sm
```

Las versiones de `requirements.txt` están fijadas. No es una precaución
decorativa: una actualización del cliente de extracción movió la ruta de sus
excepciones y rompió la ingesta sin aviso, y la reproducibilidad de las cifras
del capítulo 8 depende de poder reinstalar el mismo entorno.

Después copie `.env.example` a `.env` y ponga su token de extracción. El
archivo `.env` está excluido del repositorio.

```bash
cp .env.example .env
```

## Uso

### Ejecutar el pipeline completo

```bash
python main.py --queries "#TikTokMadeMeBuyIt" "#ProductosVirales" --items 30 --top_n 10
```

Cada publicación raspada consume créditos del servicio de extracción, así que
`--items` controla directamente el costo de la corrida. Los valores por defecto
son deliberadamente bajos.

Opciones adicionales:

| Opción | Qué hace |
|---|---|
| `--geo` | Mercado donde se detecta la tendencia. Cadena vacía significa mundial. |
| `--geo-local` | Mercado objetivo contra el que se mide el rezago. Por defecto `CO`. |
| `--no-compare-geos` | Valida solo en el mercado de detección, sin medir rezago. |

### Ver los resultados

```bash
streamlit run dashboard/app.py
```

El tablero es una capa de solo lectura. Muestra el ranking de oportunidades, la
descomposición de cada puntaje en sus tres componentes, el detalle por producto
y la procedencia del dato. Declara de forma explícita cuándo una ventana de
adopción no es medible, en lugar de mostrar un número indistinguible del azar.

### Exportar sin exponer el corpus

```bash
python exportar_resultados.py --salida resultados.csv
```

Exporta únicamente la tabla `final_trends`, con nombres de producto y métricas
agregadas. Existe porque `raw_social_data` guarda los comentarios recolectados,
y publicar el archivo `.db` como artefacto trasladaría ese material a un
almacenamiento externo.

### Ejecutar las pruebas

```bash
python ejecutar_pruebas.py
python ejecutar_pruebas.py --detalle
```

Son 130 comprobaciones repartidas en cinco suites que no consumen servicios de
pago. Una suite que termina sin imprimir su línea de resultado se cuenta como
error y no como éxito, porque un archivo de pruebas que no comprueba nada
saldría con código cero y daría una falsa sensación de cobertura. Las
comprobaciones omitidas se reportan aparte, nunca como aprobadas.

## Organización del repositorio

| Ruta | Contenido |
|---|---|
| `main.py` | Punto de entrada del pipeline. |
| `src/data_ingestion.py` | Módulo 1. Extracción y vinculación de comentarios. |
| `src/nlp_layer.py` | Módulo 2. Idioma, entidades e intención de compra. |
| `src/cross_validation.py` | Módulo 3. Validación contra el índice de búsqueda y rezago entre mercados. |
| `src/database.py` | Módulo 4. Persistencia, Opportunity Score y esquema. |
| `src/reporting.py` | Capa de lectura que alimenta el tablero y la exportación. |
| `dashboard/app.py` | Tablero de oportunidades. |
| `test_*.py` | Las cinco suites de verificación. |
| `ejecutar_pruebas.py` | Corredor que agrega el resultado de todas las suites. |
| `construir_anotacion.py` | Genera los libros de anotación y las claves maestras. |
| `construir_conciliacion.py` | Arma el libro de la sesión de conciliación. |
| `evaluar_anotacion.py` | Calcula concordancia, estándar de oro y métricas del OE2. |
| `experimento_correlacion.py` | Distribución nula por simulación para el umbral de significancia. |
| `anotacion/` | Manual de anotación, versiones 1 y 2. |
| `.github/workflows/` | Verificación continua y orquestación del pipeline. |

## Orquestación

`pruebas.yml` se ejecuta en cada cambio. Instala las dependencias, valida que
los flujos de trabajo sean sintácticamente correctos, corre las 130
comprobaciones, verifica que el tablero compile y comprueba que no se hayan
versionado libros de anotación ni bases de datos.

`pipeline.yml` ejecuta el robot. Se dispara a mano desde la pestaña Actions y
también los lunes a las 06:00 UTC, pero **la corrida programada solo procede si
existe la variable de repositorio `EJECUCION_PROGRAMADA` con valor `true`**. El
disparo automático viene desactivado de fábrica porque cada ejecución gasta
créditos, y nadie debería descubrir ese consumo después del hecho. La ejecución
manual siempre está disponible. Requiere el secreto `APIFY_API_TOKEN`.

## Qué no está en el repositorio, y por qué

No se versionan las bases de datos, los libros de anotación, las claves
maestras ni los registros de ejecución. Contienen comentarios recolectados de
plataformas públicas y, pese al filtrado de identificadores, son material
sensible bajo la Ley 1581 de 2012. La exclusión no depende de la disciplina de
quien hace el commit: la verificación continua falla si alguno de esos archivos
aparece versionado.

El sistema aplica el principio de minimización antes de persistir. Descarta los
identificadores de usuario y las secuencias numéricas susceptibles de
identificar a una persona, como los números de contacto que aparecen en los
comentarios. Ese filtro se incorporó después de encontrar, entre las entidades
extraídas, una cadena que combinaba un término comercial con un teléfono
difundido por un usuario.

## Estado del prototipo

Corresponde a un nivel de madurez tecnológica TRL 4, validado en entorno
controlado. Medido contra el corpus anotado de la segunda ronda, el extractor
de entidades alcanza 40,3 % de precisión y esa es la principal limitación del
sistema: cerca de tres de cada cinco candidatos que llegan a la validación
externa no son productos. El clasificador de intención obtiene 87,1 % de
precisión y 27,9 % de recall, cifras que el documento reporta como descripción
de la corrida y no como evidencia de desempeño generalizable, porque la
concordancia entre anotadores en esa tarea no superó el umbral de 0,70 fijado
como compuerta.
