# tex
An AI agent to help you file tax returns easily. 

Why another tax-filing app? From tech POV, every tax law change requires TurboxTax to reprogram their backend. This costs them and eventually costs end users. 

**SOLUTION:** 
The law change could be easily reflected by updates in JSON files that are consumed by AI. A normal person without coding knowledge can easily update JSON files and eventually reduces cost on end users. 

## How to run
1) Run "mlflow ui". The mlflow allows to trace errors for dev debugging.
2) Run "langgraph dev". Then, you will be redirected for a UI for user testing.

**NOTE:** above steps assume that you have [mlflow](https://mlflow.org/docs/latest/ml/tracking/) and [langgraph/langchain](https://langchain-ai.github.io/langgraph/) installed.