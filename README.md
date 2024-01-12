# StudentSurveyNaturalLanguageProcessing
As of 1/06/23 This is my current project
**Repository and File Information**
- code folder: Contains the Jupyter Notebook and all code files

- sampleData folder: Sample Data. Due to FERPA, I can not publish real student data/survey that we will train the model on. To this end, I have used chatGPT to generate data to train my model for this demo.
  
- ReadME.MD file: This file 
  
- .ipynb_checkpoints folder: Checkpoints for my Jupyter Notebook file
  
**Project Story**

- Large online courses have surveys that instructors need to manually sort through which may take hours of time, and for certain MOOCs may be borderline impossible
  
- Dr. Mayer, a professor I have researched with and my former Linear Algebra and Multivariable Calculus professor brought this problem up with me.
  
- To reduce instructor effort in large courses, this project serves to categorize survey responses using machine learning and then automate the appropriate instructor action (send an email with a response to the FAQ) or flag certain responses that require instructor attention.
  
- I am working with Dr. Mayer on this project 

**Important Dates**

- 10/31/23: First Sprint Deadline (Minimum Viable Product) ✅

- 11/6/23: Proposal Deadline ✅

- 4/15/23 -> 4/16/23: University System of Georgia Education Conference 🔵


**Current Progress**

- Sprint 1 complete, basic three-pronged classifier created with tensor flow.
  No major issues were detected in testing.
  
- Sprint 2 complete, 4-pronged classifier that combines various flows.
  Sprint 2 issues: The model has been overfitted due to majority of data being NC it started categorizing every response as NC. 
  Possible Options to fix the problem in Sprint 2 (model categorizes everything as no concern due to a large amount of no concern within the data set):

    1. Organically make my data better by adding more options that have concerns and less options that are "no concern" (This would be the best for JUST this problem BUT LACKS Generalization)
    2. Data Augmentation (maybe just cloning?) (I think this might be the best option right now as far as long term expansion goes)
    3. Changing to semi supervised learning (Not enough data)
    4. Convert to a model that uses transfer learning instead (Need to look into this more)
       
- Sprint 3 complete, Back Translation Augmentation Attempt:
    Sprint 3 issues: Major Roadblock. Google Translate API is extremely slow and unreliable. An alternative solution needs to be found.

- Sprint 4 COMPLETE, Look into Alternate Data Augmentation Method using Open AI API.
  
    Important Resource: [https://cookbook.openai.com/examples/how_to_handle_rate_limits](url)
  
    Open-AI API fine tuning problems: [https://medium.com/@abhishekmazumdar94/fine-tuning-an-open-ai-model-dc78e6ad5a07 ](url)
    THIS HAS BEEN DEEMED AS FEASIBLE. However it is time-consuming and I will get back to this after I have a viable first product. I have managed to generate augmented data.

    This method has been successfully implemented on a smaller scale.
    

  
- Sprint 5 Complete!, Use an LLM for text categorization

  Important Resources: [https://towardsdatascience.com/choosing-the-right-language-model-for-your-nlp-use-case-1288ef3c4929](url)

  SUCCESS WITH BERT
  
                       [ https://www.youtube.com/watch?v=IzbjGaYQB-U&ab_channel=PritishMishra](url)
- Sprint 6, Improve performance on more complicated cases OR Implement Unsupervised learning
  
  Important Resources: To be found


  
  Method: To be determined
  
  Possible Ideas: Augmented REAL data by using the same method as I used to generate augmented data. Reinforcement training of sorts with real data. Transfer learning?

