# Project File Inventory

This document is a repository reference for the project-authored tree. It explains what each main file and folder is for so the codebase is easier to navigate.

## Scope

1. Includes project files and folders in the repository root, Python backend, Java backend, Flutter app, data assets, scripts, tests, and existing documentation.
2. Excludes vendored or generated trees such as `flutter/`, `artifacts/`, `__pycache__/`, `.pytest_cache/`, and `mushroom_id_app/.idea/`.
3. Repeated numbered assets with the same role are grouped by exact filename range where that is clearer than repeating identical descriptions 30 times.

## Directory inventory

| Path | Purpose |
| --- | --- |
| `.` | Repository root for the full mushroom identification system. |
| `api/` | FastAPI service that exposes the 4-step identification workflow. |
| `config/` | Shared Python configuration for image-model training and evaluation. |
| `data/` | Dataset assets, data preparation utilities, and placeholder output folders. |
| `data/evaluation/` | Reserved location for generated evaluation reports and metrics. |
| `data/processed/` | Reserved location for processed ML-ready dataset exports. |
| `data/raw/` | Source dataset manifests, XML knowledge sources, and raw image assets. |
| `data/raw/images/` | Raw species image library used for model work and regression checks. |
| `data/raw/images/AM.MU/` | Fly Agaric image subset. |
| `data/raw/images/AM.VI/` | Amanita virosa image subset. |
| `data/raw/images/BO.ED/` | Porcini image subset. |
| `data/raw/images/Brunsopp/` | Brown bolete image subset for the Step 1 "Other Boletus" path. |
| `data/raw/images/CA.CI/` | Chanterelle image subset. |
| `data/raw/images/CR.CO/` | Black trumpet image subset. |
| `data/raw/images/HY.PS/` | False chanterelle image subset. |
| `Docs/` | Long-form project documentation, phase summaries, plans, and review notes. |
| `java-backend/` | Spring Boot proxy backend used by the Flutter client. |
| `java-backend/src/` | Java source root for runtime code, resources, and tests. |
| `java-backend/src/main/` | Production Java code and Spring resources. |
| `java-backend/src/main/java/` | Main Java package source root. |
| `java-backend/src/main/java/se/` | Namespace container for the Swedish domain package. |
| `java-backend/src/main/java/se/mushroomid/` | Main Spring Boot application package. |
| `java-backend/src/main/java/se/mushroomid/config/` | Spring configuration beans for CORS and HTTP clients. |
| `java-backend/src/main/java/se/mushroomid/controller/` | REST controller layer for Flutter-facing endpoints. |
| `java-backend/src/main/java/se/mushroomid/model/` | Request DTOs that mirror Python API payloads. |
| `java-backend/src/main/java/se/mushroomid/service/` | Proxy service that forwards requests to FastAPI. |
| `java-backend/src/main/resources/` | Spring Boot runtime configuration files. |
| `java-backend/src/test/` | Java test source root. |
| `java-backend/src/test/java/` | Java unit and slice test package root. |
| `java-backend/src/test/java/se/` | Namespace container for Java tests. |
| `java-backend/src/test/java/se/mushroomid/` | Root package for Java tests. |
| `java-backend/src/test/java/se/mushroomid/controller/` | Controller-layer tests for the Java proxy API. |
| `java-backend/src/test/java/se/mushroomid/service/` | Service-layer tests for the Python proxy client. |
| `models/` | Core Python identification, CV, ML, LLM, and aggregation modules. |
| `mushroom_id_app/` | Flutter client application for web and mobile. |
| `mushroom_id_app/lib/` | Main Flutter application code. |
| `mushroom_id_app/lib/models/` | Empty placeholder for future Flutter-side data models. |
| `mushroom_id_app/lib/pages/` | Flutter screens for capture, traversal, results, history, and settings. |
| `mushroom_id_app/lib/providers/` | GetX controllers for app state. |
| `mushroom_id_app/lib/services/` | Flutter services for backend access, image handling, and storage. |
| `mushroom_id_app/lib/utils/` | Flutter utility code such as translations. |
| `mushroom_id_app/lib/widgets/` | Reusable Flutter widgets. |
| `mushroom_id_app/test/` | Flutter unit, widget, and integration tests. |
| `mushroom_id_app/test/pages/` | Widget tests for individual screens. |
| `mushroom_id_app/test/providers/` | Provider/controller tests. |
| `mushroom_id_app/test/services/` | Service-level tests for API, image, and storage logic. |
| `mushroom_id_app/web/` | Flutter web shell files and PWA metadata. |
| `mushroom_id_app/web/icons/` | Web app icon assets for browser and installable-PWA use. |
| `scripts/` | Training, evaluation, and manual smoke-test scripts. |
| `tests/` | Python automated tests for API helpers and model modules. |

## File inventory

### Root files

| Path | Purpose |
| --- | --- |
| `.gitignore` | Ignores Python cache output under `scripts/__pycache__/`. |
| `Makefile` | Common commands for running the Python API, Java backend, Flutter web app, tests, and Ollama. |
| `README.md` | Top-level quick start, project overview, and high-level structure summary. |
| `requirements.txt` | Python dependency manifest for data tooling, ML, FastAPI, and tests. |
| `run_emulator.sh` | Shell helper that boots an Android emulator and launches the Flutter app. |

### FastAPI and Python configuration

| Path | Purpose |
| --- | --- |
| `api/main.py` | FastAPI entry point that wires together Step 1 image analysis, Step 2 tree traversal, Step 3 trait comparison, and Step 4 final aggregation. |
| `api/schemas.py` | Pydantic request models for Step 2-4 API payloads. |
| `api/scoring.py` | Scoring and response-adaptation helpers used by the FastAPI layer. |
| `config/image_model_config.py` | Central configuration for image-model paths, preprocessing, augmentation, and training hyperparameters. |

### Data and dataset assets

| Path | Purpose |
| --- | --- |
| `data/README.md` | Dataset guide explaining raw files, processing flow, validation, and export outputs. |
| `data/dataset_utils.py` | Dataset loader, validator, XML trait reader, and export helpers. |
| `data/evaluation/.gitkeep` | Keeps the evaluation output directory committed while it is otherwise empty. |
| `data/prepare_data.py` | Prepares training data, image processing output, and encoded trait features. |
| `data/processed/.gitkeep` | Keeps the processed-data directory committed while it is otherwise empty. |
| `data/raw/dataset_split.csv` | Train/validation/test assignments for image samples. |
| `data/raw/images/AM.MU/Amanita_muscaria_1.jpg`-`Amanita_muscaria_30.jpg` | Thirty raw Fly Agaric sample images used for dataset curation and regression checks. |
| `data/raw/images/AM.VI/Amanita_virosa_1.jpg`-`Amanita_virosa_30.jpg` | Thirty raw Amanita virosa sample images. |
| `data/raw/images/BO.ED/Boletus_edulis_1.jpg`-`Boletus_edulis_30.jpg` | Thirty raw Porcini sample images. |
| `data/raw/images/Brunsopp/Boletus_badius_1.jpg`-`Boletus_badius_30.jpg` | Thirty raw brown bolete sample images used for the "Other Boletus" path. |
| `data/raw/images/CA.CI/Cantharellus_cibarius_1.jpg`-`Cantharellus_cibarius_30.jpg` | Thirty raw Chanterelle sample images. |
| `data/raw/images/CR.CO/Craterellus_cornucopioides_1.jpg`-`Craterellus_cornucopioides_30.jpg` | Thirty raw Black Trumpet sample images. |
| `data/raw/images/HY.PS/Hygrophoropsis_aurantiaca_1.jpg`-`Hygrophoropsis_aurantiaca_30.jpg` | Thirty raw False Chanterelle sample images. |
| `data/raw/images/sources.txt` | Source-attribution ledger for downloaded image assets, mostly from iNaturalist. |
| `data/raw/key.xml` | Swedish mushroom decision tree consumed by the Step 2 traversal engine. |
| `data/raw/lookalikes.csv` | Known edible/toxic lookalike pairs with distinguishing notes and risk levels. |
| `data/raw/species.csv` | Master species list with bilingual names, edibility, toxicity, and priority lookalikes. |
| `data/raw/species_images.csv` | Image metadata catalog for dataset assets. |
| `data/raw/species_traits.csv` | Flat CSV representation of species traits by category and field. |
| `data/raw/species_traits.xml` | XML representation of species traits used by XML-based comparison code. |
| `data/validate_data.py` | CLI entry point for dataset validation checks. |

### Project documentation

| Path | Purpose |
| --- | --- |
| `Docs/01-literature-review.md` | Phase 1 literature review and requirements analysis notes. |
| `Docs/02-species-traits.md` | Trait taxonomy extracted from *Nya Svampboken*. |
| `Docs/03-dataset-construction.md` | Phase 2 dataset-construction and architecture write-up. |
| `Docs/04-system-architecture.md` | Detailed system architecture and UML-oriented design notes. |
| `Docs/05-data-dictionary.md` | Data dictionary for dataset columns and structures. |
| `Docs/06-image-recognition.md` | Phase 3 image-recognition module documentation. |
| `Docs/07-trait-classification.md` | Phase 4 trait-classification module documentation. |
| `Docs/08-llm-classification.md` | Phase 4 LLM-classification module documentation. |
| `Docs/09-hybrid-system.md` | Phase 5 hybrid-system integration documentation. |
| `Docs/CODE-REVIEW-LLM-HYBRID.md` | Review findings for the LLM and hybrid integration phases. |
| `Docs/CODE_REVIEW.md` | Review findings covering earlier implementation phases. |
| `Docs/demo_instruction` | Demo runbook for starting backend and frontend locally. |
| `Docs/EMULATOR-SETUP.md` | Android emulator setup guide for Flutter testing. |
| `Docs/implementationplan.md` | End-to-end implementation plan for the mushroom identification system. |
| `Docs/MANUAL-VERIFICATION.md` | Manual command checklist for verifying Phases 1-3. |
| `Docs/PHASE-2-SUMMARY.md` | Completion summary for Phase 2. |
| `Docs/PHASE-3-SUMMARY.md` | Completion summary for Phase 3. |
| `Docs/PHASE-4-SUMMARY.md` | Completion summary for Phase 4. |
| `Docs/PHASE-5-PLAN.md` | Initial Phase 5 plan before hybrid integration work. |
| `Docs/PHASE-5-SUMMARY.md` | Final summary for Phase 5 hybrid integration. |
| `Docs/PHASE-5-TEST-REPORT.md` | Test report for Phase 5 hybrid integration. |
| `Docs/PROJECT-FILE-INVENTORY.md` | This repository file-and-folder reference. |
| `Docs/Projektplan reviderad-4.pdf` | PDF export of the revised project plan. |
| `Docs/Projektplan reviderad.md` | Markdown version of the revised project plan. |

### Java backend

| Path | Purpose |
| --- | --- |
| `java-backend/pom.xml` | Maven project file for the Spring Boot proxy backend. |
| `java-backend/src/main/java/se/mushroomid/MushroomIdApplication.java` | Spring Boot application bootstrap class. |
| `java-backend/src/main/java/se/mushroomid/config/CorsConfig.java` | Configures backend CORS policy from application properties. |
| `java-backend/src/main/java/se/mushroomid/config/WebClientConfig.java` | Builds the `WebClient` used to call the Python API with timeout and payload limits. |
| `java-backend/src/main/java/se/mushroomid/controller/IdentificationController.java` | Flutter-facing REST controller that mirrors the full 4-step pipeline under `/api`. |
| `java-backend/src/main/java/se/mushroomid/model/Step2AnswerRequest.java` | DTO for Step 2 answer submissions. |
| `java-backend/src/main/java/se/mushroomid/model/Step2StartRequest.java` | DTO for starting Step 2 tree traversal with visible traits. |
| `java-backend/src/main/java/se/mushroomid/model/Step3CompareRequest.java` | DTO for Step 3 trait comparison requests. |
| `java-backend/src/main/java/se/mushroomid/model/Step4FinalizeRequest.java` | DTO for Step 4 final aggregation requests. |
| `java-backend/src/main/java/se/mushroomid/service/PythonApiService.java` | Proxy service that forwards Flutter requests to the FastAPI backend and returns raw JSON. |
| `java-backend/src/main/resources/application.properties` | Runtime settings for ports, upload limits, timeout, Python base URL, and CORS. |
| `java-backend/src/test/java/se/mushroomid/controller/IdentificationControllerTest.java` | Tests controller routing and request handling. |
| `java-backend/src/test/java/se/mushroomid/service/PythonApiServiceTest.java` | Tests the Java proxy service's interactions with the Python API client. |

### Python model modules

| Path | Purpose |
| --- | --- |
| `models/__init__.py` | Package marker for Python model modules. |
| `models/cnn_classifier.py` | EfficientNet-B3 image classifier that uses fine-tuned weights when available. |
| `models/final_aggregator.py` | Step 4 aggregator that combines Step 1-3 outputs into the final recommendation payload. |
| `models/hybrid_classifier.py` | Hybrid multi-method classifier with confidence aggregation, safety warnings, and lookalike logic. |
| `models/image_processor.py` | Image preprocessing and augmentation utilities for training and inference workflows. |
| `models/image_recognition.py` | Transfer-learning image-recognition model builder for TensorFlow or PyTorch experiments. |
| `models/key_tree_traversal.py` | Step 2 engine that parses `key.xml`, auto-answers from visible traits, and manages traversal sessions. |
| `models/llm_classifier.py` | LLM-based classifier with prompt templates, species database, and multiple backends. |
| `models/observation_parser.py` | Natural-language parser that converts free-text observations into structured traits. |
| `models/trait_classifier.py` | Scikit-learn trait classifier supporting decision tree and random forest training. |
| `models/trait_database_comparator.py` | Step 3 comparator that matches visible traits against stored species profiles and lookalikes. |
| `models/trait_processor.py` | Trait encoding and feature-engineering utilities for ML-ready tabular trait vectors. |
| `models/visual_trait_extractor.py` | Step 1 computer-vision trait extractor and classical image-based species scorer. |

### Flutter application

| Path | Purpose |
| --- | --- |
| `mushroom_id_app/.gitignore` | Flutter app ignore rules for build and tool output. |
| `mushroom_id_app/.metadata` | Flutter tool metadata for the app project. |
| `mushroom_id_app/analysis_options.yaml` | Dart analyzer and lint configuration. |
| `mushroom_id_app/DEPLOYMENT.md` | Flutter deployment guide for mobile builds and distribution. |
| `mushroom_id_app/lib/main.dart` | Flutter app bootstrap, theme setup, route registration, and home page. |
| `mushroom_id_app/lib/pages/camera_page.dart` | Image capture/upload screen with preview, rotation, and Step 1 handoff. |
| `mushroom_id_app/lib/pages/history_page.dart` | History screen and detail view for saved identifications. |
| `mushroom_id_app/lib/pages/questionnaire_page.dart` | Legacy manual trait questionnaire flow. |
| `mushroom_id_app/lib/pages/results_page.dart` | Results screen that normalizes and renders final identification output. |
| `mushroom_id_app/lib/pages/settings_page.dart` | Settings screen for preferences, API URL, and debug toggles. |
| `mushroom_id_app/lib/pages/tree_traversal_page.dart` | Step 2 question-and-answer UI for the `key.xml` traversal flow. |
| `mushroom_id_app/lib/providers/history_provider.dart` | GetX controller for loading, saving, filtering, and deleting history entries. |
| `mushroom_id_app/lib/providers/identification_provider.dart` | GetX controller that orchestrates the full 4-step identification pipeline. |
| `mushroom_id_app/lib/providers/language_provider.dart` | Locale state controller for English/Swedish switching. |
| `mushroom_id_app/lib/services/identification_api_service.dart` | Flutter HTTP client with backend auto-detection and Step 1-4 request methods. |
| `mushroom_id_app/lib/services/image_service.dart` | Local image validation and size-formatting helper. |
| `mushroom_id_app/lib/services/storage_service.dart` | History/preference persistence layer using SharedPreferences on web and SQLite elsewhere. |
| `mushroom_id_app/lib/utils/app_translations.dart` | Localized UI strings for English and Swedish. |
| `mushroom_id_app/lib/widgets/language_flag_button.dart` | App bar widget for quick locale switching. |
| `mushroom_id_app/mushroom_identification.iml` | IntelliJ/Android Studio module metadata file. |
| `mushroom_id_app/pubspec.lock` | Locked Flutter dependency versions. |
| `mushroom_id_app/pubspec.yaml` | Flutter package manifest and dependency list. |
| `mushroom_id_app/README.md` | Default Flutter starter README for the app subproject. |
| `mushroom_id_app/TESTING-GUIDE.md` | Flutter testing guide covering unit, widget, and integration tests. |
| `mushroom_id_app/test/integration_test.dart` | End-to-end Flutter integration test entry point. |
| `mushroom_id_app/test/pages/camera_page_test.dart` | Widget tests for the camera page. |
| `mushroom_id_app/test/pages/history_page_test.dart` | Widget tests for the history page. |
| `mushroom_id_app/test/pages/questionnaire_page_test.dart` | Widget tests for the questionnaire page. |
| `mushroom_id_app/test/pages/results_page_test.dart` | Widget tests for the results page. |
| `mushroom_id_app/test/pages/settings_page_test.dart` | Widget tests for the settings page. |
| `mushroom_id_app/test/pages/tree_traversal_page_test.dart` | Widget tests for the tree-traversal page. |
| `mushroom_id_app/test/providers/history_provider_test.dart` | Tests for history-provider state and helpers. |
| `mushroom_id_app/test/providers/identification_provider_test.dart` | Tests for the main identification provider. |
| `mushroom_id_app/test/providers/language_provider_test.dart` | Tests for locale toggling and persistence behavior. |
| `mushroom_id_app/test/services/identification_api_service_test.dart` | Tests for backend-resolution and API-client behavior. |
| `mushroom_id_app/test/services/image_service_test.dart` | Tests for image validation helpers. |
| `mushroom_id_app/test/services/storage_service_test.dart` | Tests for storage serialization and persistence logic. |
| `mushroom_id_app/test/widget_test.dart` | Default top-level Flutter widget test scaffold. |
| `mushroom_id_app/web/favicon.png` | Browser favicon for the web app shell. |
| `mushroom_id_app/web/icons/Icon-192.png` | 192px PWA icon asset. |
| `mushroom_id_app/web/icons/Icon-512.png` | 512px PWA icon asset. |
| `mushroom_id_app/web/icons/Icon-maskable-192.png` | 192px maskable PWA icon asset. |
| `mushroom_id_app/web/icons/Icon-maskable-512.png` | 512px maskable PWA icon asset. |
| `mushroom_id_app/web/index.html` | Flutter web host page and bootstrap shell. |
| `mushroom_id_app/web/manifest.json` | Web app manifest for installability and icon metadata. |

### Training, evaluation, and manual scripts

| Path | Purpose |
| --- | --- |
| `scripts/evaluate_image_model.py` | Evaluates saved image-model artifacts and reports metrics. |
| `scripts/evaluate_llm_model.py` | Evaluates the LLM classifier and compares it with trait-based output. |
| `scripts/evaluate_trait_model.py` | Evaluates trained trait-classification models and compares algorithms. |
| `scripts/test_cli.py` | Manual CLI client for probing backend health and running identification requests. |
| `scripts/test_hybrid_system.py` | Standalone script for exercising aggregation strategies, lookalikes, and safety warnings. |
| `scripts/train_cnn.py` | Training script for the EfficientNet-based 7-species CNN classifier. |
| `scripts/train_image_model.py` | General training entry point for transfer-learning image models. |
| `scripts/train_llm_model.py` | Validation and training-oriented checks for the LLM classifier and parser. |
| `scripts/train_trait_model.py` | Training entry point for trait-based classifiers. |

### Python tests

| Path | Purpose |
| --- | --- |
| `tests/__init__.py` | Package marker for Python tests. |
| `tests/test_final_aggregator.py` | Unit tests for Step 4 aggregation logic and verdict handling. |
| `tests/test_hybrid_classifier.py` | Unit tests for hybrid aggregation methods and result structure. |
| `tests/test_key_tree_traversal.py` | Unit tests for Step 2 tree traversal sessions and auto-answering. |
| `tests/test_scoring.py` | Unit tests for API scoring helpers and response adaptation. |
| `tests/test_trait_regression_real_images.py` | Regression tests that check real-image trait signals across dataset subsets. |
| `tests/test_visual_trait_extractor.py` | Unit tests for Step 1 visual trait extraction and synthetic-image scoring. |
