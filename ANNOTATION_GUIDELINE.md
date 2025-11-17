# OKE Dataset Annotation Guideline  
  
his document is prepared for OKE dataset annotators. Twelve different OPC UA companion specifications, technical documents were utilized for the OKE Dataset. These include PackML, Robotics, PROFINET, Machine Tools, Weihenstephan, IOLink, FX, Machine Vision CCM[^1], Machine Vision AMCM[^2], AutoID, ISA95, and PADIM. Each document will be annotated by two annotators independently, except for stabilizing the input. The following steps should be followed:  
  
In order to get the PDFs of these companion specifications, please visit [OPC Foundation Developer Tools](https://opcfoundation.org/developer-tools/documents).  
  
[^1]: Control, configuration management, recipe management, result management  
[^2]: Asset management and condition monitoring  
  
## Step 1: Sentence Extraction  
- **Sentence Extraction Tool**: Use the internal sentence extraction tool. Those who do not have access to this tool can manually extract sentences.  
- **Sentence Review**: Both annotators review the sentences together and correct any errors.  
  
## Step 2: Manual Process  
The resulting CSV file should be reviewed and corrected for the following issues:  
1. **Incorrectly Separated or Incomplete Sentences**: The correct sentence should be completed by mutual agreement.  
2. **Unseparated Adjacent Sentences**: Structures containing multiple sentences should be separated into individual sentences.  
3. **Coreference Resolution**: If there is ambiguity due to pronouns and it has not been automatically resolved, it should be manually resolved.  
4. **CSV File Format**: Each document is represented by a CSV file. The first row contains column names: `Sentence Number`, `Sentence`, `Page`, `Rule Sentence (y/n)`, `IM-Rule Sentence (y/n)`, `Information Model Keywords`, `Links`, `Constraint Keywords`, `Relation Keywords`, `Numbers`, `Quotes`, `Runtime only`. The first three columns should be filled, and the other columns should be empty for each row. Any errors should be manually corrected.  
  
## Step 3: Annotation  
Three types of annotation will be performed:  
1. **Sentence Classification**: Each sentence is classified by marking two columns. The definitions and example sentences can be seen in Table 1:  
  ##### Table 1: Defintions of Sentence Classification Categories  
  
| **Class Category**                                                                 | **Definition**                                                                                       | **Example from the Companion Specifications**                                                                 |  
|------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|  
| **Rule Sent. Type 1: Information-model-rule sentence**                             | A sentence within the companion specification indicates a rule pertaining to the structure of the information model. | StopReasonExtent defines the maximum number of stop reason elements or available.                             |  
| **Rule Sent. Type 2: Run-time-rule sentence**                                      | A sentence within the companion specification indicates a rule that can only be checked at run-time. | Any numeration must include "Produce" as enumeration 1.                                                      |  
| **Non-rule sentence**                                                              | All other sentences within the companion specification that do not contain a rule sentence affecting the information model or the operation at run-time. | Figure 10 provides an overview of the instance object model for PackML. 

   - **Rule Sentence (y/n)**: Indicates whether the sentence is a rule.  
   - **IM-Rule Sentence (y/n)**: If the first category is "y", the second can be "y" or "n". If the first category is "n", the second will automatically be "n".  

1. **Named Entity Annotation**: There are six categories defined by us. Examples and definitions can be seen in Table 2. Accordingly, it is expected that words or groups of words in the sentence will be marked.  

##### Table 2: Definitions of Named Entity Categories  
  
| **Entity Category**    | **Definition**                                                                                     | **Example from the sentences from Comp. Specs.**               |  
|------------------------|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------|  
| **Information Model**  | A word or group of words within the sentence corresponds to the label of a node in the relevant information models. | StopReasonExtend, enumeration, etc.                            |  
| **Constraint**         | Indicates a word or phrase that denotes a constraint in a sentence.                                | must, at least, any, etc.                                      |  
| **Relation**           | Represents the relationship between entities.                                                      | has component, include, etc.                                   |  
| **Quotation**          | All the words or groups of words in quotation marks.                                               | "Produce", 'devices', etc.                                     |  
| **Number**             | Represents all numbers written in digits and words.                                                | 12, -123, twenty-two, fifty six, 0, etc.                       |  
| **Runtime-only**       | Indicates a run-time state.                                                                        | start, stop, assign, etc.                                      |  
   - **Rules**:  
     - Words or phrases should be tagged in only one entity category.  
     - Entities should be marked as they appear in the sentence, without changing case or other modifications.  
     - Entities should be separated by commas.  
     - Sentences within quotation marks should include the quotation marks and be represented with double quotes (").  
   - **Challenges**:  
     - The same word may be used in different senses and may need to be placed in different categories.  
     - The same or similar meaning words in different sentences should be marked in the same category.  
     - The same word may be used differently in different companion specifications.  
     - Different annotators may annotate the same thing differently.  
3. **Entity Linking**: Ensure that entities are correctly linked.  
##### Table 3: Definitions of Entity Linking  
  
| **Category**    | **Definition**                                                                                                                                                                                                                                                    | **Example**                                                                                     |  
|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|  
| **Entity Link** | Every entity under the Information Model category should correspond to a node within the information model (or ontology). The link of the relevant information model entity in the model is called the entity link.                                               | **IM Entity:** StopReasonExtent, object types, etc. <br> **Entity Link:** packML:StopReason, opcua:ObjectTypes |  
  
## Step 4: Data Analysis  
- **Inter Annotator Agreement (IAA)** is calculated, and data analysis is conducted. Data analysis has three important effects:  
  1. An annotator can see if they have marked the same entity in different categories within the same document.  
  2. An annotator can see how they have annotated the same entity in different companion specifications and intervene in case of conflict.  
  3. IAA is obtained for conflicts between different annotators, and a consensus is reached.  
  
## Step 5: Dataset Release  
After reaching a consensus among annotators, the 12 CSV files are combined and published.  
  
By following these steps, annotators can accurately and consistently annotate the sentences.  

## Author
Nilay Tufek Özkaya
ntufek@gmail.com