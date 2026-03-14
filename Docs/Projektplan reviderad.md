# **AI-Based Mushroom Identification Using Image Recognition and Trait-Based Classification**

## **Abstract**

Mushroom identification is a challenging task due to the large number of species and the visual similarity between edible and poisonous mushrooms. Traditional identification methods rely on field guides and expert knowledge, which can be difficult for beginners to use. Recent advances in artificial intelligence and smartphone technology make it possible to develop systems that assist users in identifying species using images and structured observations.

This thesis investigates the use of artificial intelligence methods for mushroom identification in a smartphone application context. The project will develop a prototype system that combines image recognition with trait-based classification derived from observable mushroom characteristics. In addition, a large language model (LLM) will be used to interpret textual descriptions generated from user observations.

The study will compare the effectiveness of different identification approaches, including image-based recognition, trait-based machine learning, and LLM-based reasoning. The project will also investigate whether combining image recognition with structured trait information improves identification accuracy. The performance of the different methods will be evaluated using classification metrics and experiments with incomplete or uncertain input data.

The goal of the project is both to develop a functional prototype and to analyze how different AI techniques can support biological species identification.

---

# **Background**

Mushroom identification has traditionally relied on field guides and expert knowledge. These resources require significant experience and can be difficult for beginners to use effectively. Many mushroom species share similar visual characteristics, and misidentification may have serious consequences if poisonous species are mistaken for edible ones.

Recent developments in smartphone technology and artificial intelligence have enabled new tools for species identification. Modern smartphones provide built-in cameras and computational resources that allow users to capture images and interact with identification systems directly in the field.

Image recognition using convolutional neural networks has proven effective for many visual classification tasks. However, visual appearance alone is often insufficient for accurate mushroom identification. In traditional mycology, species are identified using combinations of morphological traits such as cap shape, gill structure, habitat, and season.

Combining image recognition with structured biological traits may therefore improve identification accuracy. In addition, recent advances in large language models (LLMs) make it possible to interpret natural language descriptions of organisms and generate classification suggestions based on textual input.

This project aims to explore how these different approaches can be used together in a system for a mushroom identification system.

---

# **Research Questions**

The thesis investigates how different artificial intelligence approaches can support mushroom identification.

The primary research question is:

**How does combining trait-based machine learning and LLM-based classification affect mushroom identification accuracy compared to standalone methods?**

To address this question, the following sub-questions will be explored:

1. What level of identification accuracy can be achieved using image-based recognition alone?  
2. How does the performance of a trait-based machine learning model compare to an LLM-based approach that interprets textual descriptions of mushroom characteristics?  
3. How does combining image recognition with trait-based classification affect identification accuracy?  
4. How robust are the identification approaches when the provided characteristics are incomplete or ambiguous?  
5. How does the combination of LLM and machine learning methods compare in accuracy compared to standalone methods?

The answers to these questions will provide insights into how modern AI techniques can support biological classification tasks.

---

# 

# **Description of the Task**

The goal of the project is to design and implement a prototype system that assists users in identifying mushrooms using a smartphone application.

The project consists of several main components.

### **Literature Review and Problem Analysis**

A review of existing research and tools for species identification will be conducted. This includes studies on image recognition, biological classification systems, and the use of machine learning for species identification.

The review will also examine traditional mushroom identification methods and common morphological characteristics used in mycology.

## **Dataset Construction**

The dataset used in the project will be derived from the field guide *Nya Svampboken* by Pelle Holmberg. The book contains detailed descriptions of mushroom species, including morphological characteristics, habitat information, and seasonal occurrence.

These descriptions will be structured into a dataset containing traits such as:

●       cap shape

●       cap color

●       gill structure

●       stem characteristics

●       habitat

●       growth pattern

●       season

The system will focus on a limited subset of common mushroom species to ensure manageable model complexity and sufficient data per species. *Nya Svampboken* is particularly suitable for this purpose because it focuses on common edible mushrooms as well as poisonous species that are frequently mistaken for edible ones.

### **System Design**

The system architecture will consist of:

●       a smartphone application for capturing images and entering observations

●       an identification module based on image recognition

●       a trait-based classification component

●       an LLM-based module for interpreting textual descriptions

The system will be documented using UML diagrams, including use case diagrams, class diagrams, and sequence diagrams.

## **Application Development**

The prototype application will allow users to:

●       capture a photo of a mushroom

●       answer questions about observable traits

●       receive a list of possible species with confidence scores

●       view possible look-alike species

Image recognition will provide an initial set of candidate species based on visual features extracted from photographs. Trait-based classification will then refine the prediction using additional information provided by the user.

The system will also generate a ranked list of possible species when classification uncertainty is high.

## **Evaluation**

The project will experimentally evaluate the different identification approaches and analyze their effectiveness.

The evaluation will focus on:

●       classification accuracy

●       robustness to incomplete input data

●       comparison between identification methods

●       the effect of combining visual and trait-based information

The results will be analyzed and discussed in the thesis report.

---

# **Methods**

The project will follow an iterative software development process consisting of requirements analysis, system design, implementation, and evaluation.

## **Tools and Technologies**

The following technologies will be used:

●       smartphone development framework (Flutter or native Android)

●       Machine learning frameworks such as PyTorch or TensorFlow

●       Pretrained convolutional neural networks for image recognition

●       Large language models such as GPT or Llama accessed through an API

●       Git for version control

UML modeling tools will be used to document the system architecture.

---

## **Image Recognition**

Image recognition will be implemented using transfer learning with a pretrained convolutional neural network such as MobileNet, EfficientNet, or ResNet.

The pretrained model will be adapted to the mushroom dataset by training a new classification layer on mushroom images. The model will produce a ranked list of candidate species (top-k predictions) with associated probabilities.

## **Trait-Based Machine Learning**

A trait-based classification model will be trained using structured mushroom characteristics extracted from the dataset. Possible algorithms include decision trees or random forests.

These models are well suited to categorical biological features and resemble traditional identification keys used in mycology.

## **LLM-Based Classification**

A large language model such as GPT or Llama will be used through an API to interpret textual descriptions of mushroom characteristics.

User observations will be converted into structured descriptions which will be provided as input to the language model. The model will then generate a prediction of the most likely species.

## **Hybrid Identification**

The system will combine image recognition with trait-based classification.

Image recognition will provide an initial set of candidate species, while the trait-based model will refine the prediction based on additional observations. This hybrid approach aims to improve identification accuracy compared to using either method independently.

---

## **Confidence Estimation**

## The system will estimate the confidence of each prediction using classification probabilities produced by the machine learning model.

## The top-k most likely species will be presented to the user together with confidence levels. When confidence is low, the system will present alternative species and known look-alike mushrooms that may be confused with the predicted species.

## **Analytical Evaluation**

## The experimental evaluation will analyze the effectiveness of different identification approaches. The analysis will focus on three aspects:

1. ## Comparison of identification methods using metrics such as accuracy, precision, recall, and F1-score.

2. ## Robustness to incomplete or incorrect input data.

3. ## The effect of combining image recognition with trait-based classification.

## Results will be presented using tables and visualizations.

# **Relevant Courses**

## The project builds upon knowledge from several courses in computer science and software engineering, including:

## ●       Software Engineering

## ●       Database Systems

## ●       Data Structures and Algorithms

## ●       Artificial Intelligence

## ●       Human–Computer Interaction

## ●       Machine Learning

## These courses provide the theoretical and practical foundation for implementing and evaluating the system.

# **Delimitations**

## To ensure that the project remains feasible within the available timeframe, several limitations are defined.

## The system will focus on a limited subset of common mushroom species rather than attempting to cover the full diversity of mushrooms.

## Image recognition will rely on pretrained neural networks using transfer learning rather than training a model from scratch.

## The application will not be intended for safety-critical use. Clear disclaimers will inform users that the system should not be relied upon when determining whether mushrooms are safe to consume.

## Potential extensions if time permits include:

## ●       expanding the number of mushroom species

## ●       improving the machine learning model

## ●       integrating additional datasets

## ●       improving the user interface and visualization features

## If time becomes limited, the LLM-based evaluation may be simplified or restricted to a smaller dataset.

# **Time Plan**

| Period | Activity |
| ----- | ----- |
| Weeks 1–2 | Literature review and requirements analysis |
| Weeks 2–3 | Dataset construction and system architecture design |
| Weeks 3–5 | Implementation of image recognition module |
| Weeks 5–7 | Development of trait-based classification |
| Weeks 7–8 | Integration of LLM-based identification |
| Weeks 8–9 | Experimental evaluation and testing |
| Weeks 9–10 | Analysis of results |
| Weeks 10–11 | Writing and finalizing the thesis report |

## Writing of the thesis report will occur continuously throughout the project.

## Regular meetings with the project supervisor will be scheduled throughout the project.

# **References**

## Bonnet, P. et al. (2020). *Citizen science and biodiversity monitoring*. Biological Conservation.

## Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning*. MIT Press.

## Russell, S., Norvig, P. (2021). *Artificial Intelligence: A Modern Approach*. Pearson.

## Holmberg, P., & Marklund, H. (2019). *Nya Svampboken*. Bonnier Fakta.

Lee, J. J., Aime, M. C., Rajwa, B., & Bae, E. (2022). Machine learning-based classification of mushrooms using a smartphone application.

Chaschatzis, C., Karaiskou, C., Iakovidou, C., Radoglou-Grammatikis, P., Bibi, S., Goudos, S. K., & Sarigiannidis, P. G. (2025). Detection of wild mushrooms using machine learning and computer vision.