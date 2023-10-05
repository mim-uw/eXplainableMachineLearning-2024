# eXplainable Machine Learning / Wyjaśnialne Uczenie Maszynowe - 2024

[**eXplainable Machine Learning**](https://usosweb.uw.edu.pl/kontroler.php?_action=katalog2/przedmioty/pokazZajecia&zaj_cyk_id=514785&gr_nr=1) course for Machine Learning (MSc) studies at the University of Warsaw. 

Winter semester 2023/24 [@pbiecek](https://github.com/pbiecek) [@hbaniecki](https://github.com/hbaniecki)


## Meetings

Plan for the winter semester 2023/2024. MIM_UW classes are on Fridays. 

* 2023-10-06  --  Introduction, [Slides](https://htmlpreview.github.io/?https://raw.githubusercontent.com/mim-uw/eXplainableMachineLearning-2024/main/Lectures/01_introduction.html)
* 2023-10-13  --  SHAP and friends	
* 2023-10-20  --  LIME and friends	
* 2023-10-27  --  PDP and friends		
* 2023-11-03  --  PROJECT: **First checkpoint** 
* 2023-11-10  --  VIP / MCR	
* 2023-11-17  --  Fairness	
* 2023-11-24  --  Explanations specific to neural networks	
* 2023-12-01  --  Evaluation of explanations	
* 2023-12-08  --  PROJECT: **Second checkpoint** 	
* 2023-12-15  --  	
* 2024-01-12  --  	
* 2024-01-19  --  	
* 2024-01-26  -- PROJECT: **Final presentation**  	

## How to get a good grade

The final grade is based on activity in four areas:

* mandatory: Project (0-35)
* mandatory: Exam  (0-35)
* optional: Homeworks (0-24)
* optional: Presentation (0-6)

In total you can get from 0 to 100 points. 51 points are needed to pass this course.

Grades:

* 51-60: (3) dst
* 61-70: (3.5) dst+
* 71-80: (4) db
* 81-90: (4.5) db+
* 91-100: (5) bdb

## Homeworks (0-24 points)

 - [Homework 1](https://github.com/mim-uw/eXplainableMachineLearning-2024/tree/main/Homeworks/HW1)  for 0-4 points. **Deadline: 2023-10-12**
 - [Homework 2](https://github.com/mim-uw/eXplainableMachineLearning-2024/tree/main/Homeworks/HW2)  for 0-4 points. **Deadline: 2023-10-19** 
 - [Homework 3](https://github.com/mim-uw/eXplainableMachineLearning-2024/tree/main/Homeworks/HW3)  for 0-4 points. **Deadline: 2023-10-26**
 - [Homework 4](https://github.com/mim-uw/eXplainableMachineLearning-2024/tree/main/Homeworks/HW4)  for 0-4 points. **Deadline: 2023-11-09**
 - [Homework 5](https://github.com/mim-uw/eXplainableMachineLearning-2024/tree/main/Homeworks/HW5)  for 0-4 points. **Deadline: 2023-11-16**
 - [Homework 6](https://github.com/mim-uw/eXplainableMachineLearning-2024/tree/main/Homeworks/HW6)  for 0-4 points. **Deadline: 2023-11-23**

## Project (0-35 points)

This year's project involves conducting a vulnerability analysis of a predictive models using XAI tools.
This analysis should be carried out for a selected model and the results should be summarised in a short RedTeaming report.

### Key points:

- Projects can be done in groups of 1, 2 or 3 students 
- One model can be analysed by multiple groups (but the discovered vulnerabilities must not be repeated)
- The harder the project, the easier it is to obtain a higher grade.

### Important dates:

- 2023-11-03 – First checkpoint: Students chose the model, create a plan of work (to be discussed at the classes). Deliverables: 3 min presentation based on one slide.
- 2023-12-08 – Second checkpoint: Provide initial experimental results. At least one vulnerability should have been found by now.
- 2023-01-26 - Final checkpoint: Presentation of all identified  vulnerabilities

### Models:

RedTeaming analysis should be carried out for a selected model. Depending on the difficulty of the model, you may receive more or less points

- the analysis concerns your own model (e.g. from homework). Point conversion rate: **x0.8**
- the analysis applies to a model from Hugging Face (one of the models available there for any modality). Points conversion factor: **x1**
- the analysis applies to one of the popular foundation models (e.g. TabPFN https://arxiv.org/abs/2207.01848, Segment Anything https://arxiv.org/abs/2304.02643, Llama 2 https://arxiv.org/abs/2307.09288, RETFound https://www.nature.com/articles/s41586-023-06555-x). Points conversion factor: **x1.25**

### RedTeam Report:

Examples of directions to look for vulnerability (creativity will be appreciated)

- bias / fairness. Does the model discriminate against a protected attribute? (e.g. https://arxiv.org/abs/2105.02317)
- using XAI to find instance level artifacts (e.g. Clever Hans https://doi.org/10.1016/j.inffus.2021.07.015, wolf/snow https://arxiv.org/abs/1602.04938), large residuals and explanations why predictions were wrong
- unintended memorisation (e.g. https://arxiv.org/abs/1802.08232)
- drift in performance gap between datasets (e.g. another image or text dataset)
- model malfunction due to an unintendent use
- wrong behaviour on out-of-distribution samples

The final report will be a short (up to 4 pages) paper in the JMLR template. See [an example](https://github.com/mim-uw/eXplainableMachineLearning-2023/blob/main/Projects/PawelPawlik_MichalSiennicki_AlicjaZiarko/checkpoint3/report.pdf).


## Literature

We recommend to dive deep into the following books and explore their references on a particular topic of interest:

* [Explanatory Model Analysis. Explore, Explain and Examine Predictive Models](https://pbiecek.github.io/ema/) by Przemysław Biecek, Tomasz Burzykowski
* [Fairness and Machine Learning: Limitations and Opportunities](https://fairmlbook.org/) by Solon Barocas, Moritz Hardt, Arvind Narayanan
* [Interpretable Machine Learning. A Guide for Making Black Box Models Explainable](https://christophm.github.io/interpretable-ml-book/) by Christoph Molnar

