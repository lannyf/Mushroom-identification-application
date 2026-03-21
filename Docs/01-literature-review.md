# Phase 1: Literature Review and Requirements Analysis

## Objective
Conduct a comprehensive literature review on image recognition, biological classification systems, machine learning for species identification, and traditional mushroom identification methods. Document requirements for the mushroom identification system.

## Literature Review Sections

### 1. Image Recognition and Deep Learning
**Key Topics:**
- Convolutional Neural Networks (CNNs) for image classification
- Transfer learning and pretrained models (MobileNet, EfficientNet, ResNet)
- Fine-tuning techniques for domain-specific tasks
- Image classification metrics and evaluation

**Relevant Papers/Resources:**
- Goodfellow et al. (2016) - Deep Learning: Comprehensive overview
- MobileNet, EfficientNet, ResNet architecture papers
- Transfer learning best practices

**Status:** [ ] In progress

### 2. Machine Learning for Biological Classification
**Key Topics:**
- Decision trees and random forests for categorical data
- Ensemble methods in biodiversity classification
- Feature importance and interpretability
- Handling imbalanced biological datasets

**Relevant Papers/Resources:**
- Scikit-learn documentation and examples
- Chaschatzis et al. (2025) - Machine learning for wild mushroom detection
- Lee et al. (2022) - Smartphone-based mushroom classification

**Status:** [ ] In progress

### 3. Large Language Models for Species Identification
**Key Topics:**
- LLM capabilities for domain-specific classification
- Prompt engineering for biological data
- API-based LLM integration (GPT, Llama)
- Chain-of-thought reasoning for biological inference

**Relevant Papers/Resources:**
- OpenAI GPT documentation
- Few-shot learning with LLMs
- Structured prompt design

**Status:** [ ] In progress

### 4. Traditional Mycology and Mushroom Identification
**Key Topics:**
- Morphological characteristics used in mycology
- Traditional identification keys and field guides
- Edible vs. poisonous mushroom traits
- "Nya Svampboken" field guide analysis

**Relevant Papers/Resources:**
- Holmberg & Marklund (2019) - Nya Svampboken
- Bonnet et al. (2020) - Citizen science and biodiversity monitoring
- Traditional mycological classification methods

**Status:** [ ] In progress

### 5. Smartphone-Based Species Identification Systems
**Key Topics:**
- Mobile computing constraints and optimization
- User interaction patterns for field identification
- Mobile app frameworks (Flutter, native Android)
- User experience for non-expert users

**Relevant Papers/Resources:**
- Lee et al. (2022) - Smartphone application for mushroom classification
- Mobile HCI literature
- Field usability studies

**Status:** [ ] In progress

## Requirements Analysis

### Functional Requirements
1. **Image Recognition**
   - Capture mushroom photos via smartphone camera
   - Process images and return top-k species predictions
   - Provide confidence scores

2. **Trait-Based Classification**
   - Present questions about observable mushroom characteristics
   - Accept user input on traits (cap shape, color, gills, stem, habitat, growth, season)
   - Generate species predictions based on traits

3. **LLM-Based Classification**
   - Convert user observations to natural language descriptions
   - Query LLM API for species prediction
   - Extract and rank predicted species

4. **Hybrid System**
   - Combine predictions from all three methods
   - Aggregate confidence scores
   - Present ranked species list with look-alikes
   - Show identification explanation/reasoning

5. **User Interface**
   - Photo capture and gallery selection
   - Trait questionnaire with optional fields
   - Results display with confidence levels
   - Look-alike warnings and additional information

### Non-Functional Requirements
1. **Performance**
   - Image processing: < 5 seconds for prediction
   - LLM API latency acceptable for interactive use
   - Model inference on mobile device or cloud

2. **Accuracy**
   - Target: > 70% top-1 accuracy on test species
   - Robustness to incomplete trait data
   - Balanced performance across species

3. **Usability**
   - Clear, simple interface for non-experts
   - Offline capability for image recognition preferred
   - Educational value in explaining predictions

4. **Safety & Ethics**
   - Prominent disclaimer: not for safety-critical decisions
   - Educational use only
   - No reliance for edibility determination

### Data Requirements
1. **Mushroom Dataset**
   - 20-50 common mushroom species from "Nya Svampboken"
   - Structured trait data (cap shape, color, gills, stem, habitat, season, growth)
   - Training images per species (or access to mushroom image datasets)
   - Look-alike relationships for safety

2. **Training Data Distribution**
   - Balanced representation across species
   - Multiple image angles and lighting conditions
   - Varied user trait descriptions for LLM training

### Constraints & Delimitations
1. **Scope**
   - Limited to subset of common species (20-50)
   - Focus on species from Swedish field guide
   - Prototype/research system, not production

2. **Technical**
   - Use transfer learning (no training from scratch)
   - Rely on pretrained models and APIs
   - Mobile app or web/CLI interface for testing

3. **Data Handling**
   - Proper attribution of "Nya Svampboken" material
   - Respect intellectual property rights
   - Clear disclaimers on system limitations

## Key Research Questions (from project plan)

1. What level of accuracy can image recognition alone achieve?
2. How does trait-based ML compare to LLM-based approaches?
3. How does combining image + traits affect accuracy?
4. How robust are methods with incomplete/ambiguous data?
5. How does the combination of LLM + ML compare to standalone methods?

## Document Structure for Requirements Document

1. Executive Summary
2. System Overview and Context
3. Functional Requirements (organized by module)
4. Non-Functional Requirements
5. Data Requirements
6. Use Cases
7. Constraints and Assumptions
8. Success Criteria and Metrics

## Next Steps

- [ ] Gather literature sources and create annotated bibliography
- [ ] Write detailed requirements document
- [ ] Create requirements traceability matrix
- [ ] Document approved requirements for architecture phase

---

**Status:** In Progress  
**Created:** 2026-03-14  
**Last Updated:** 2026-03-14
