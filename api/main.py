from fastapi import FastAPI,HTTPException
from fastapi.responses import JSONResponse,HTMLResponse
from tempfile import NamedTemporaryFile
import pandas as pd
import numpy as np
import lightgbm
from contextlib import asynccontextmanager
from pydantic import BaseModel,field_validator,ValidationError
from evidently import Dataset,DataDefinition
from evidently import Report,BinaryClassification
from evidently.presets import ClassificationPreset

#=================================================================================
# Loading datasets and model
#=================================================================================
session={}
@asynccontextmanager
async def lifespan(app : FastAPI):
    session['model']=lightgbm.Booster(model_file='/api_img/req_data/model.txt')
    session['lookup']=pd.read_parquet('/api_img/req_data/lookup.parquet')
    session['vectors']=pd.read_parquet('/api_img/req_data/test_vectors.parquet')
    session['truth']=pd.read_parquet('/api_img/req_data/test_truth.parquet')
    yield
    session.clear()

app = FastAPI(lifespan=lifespan)

#=================================================================================
# Pydantic user input verification
#=================================================================================

class validate_idx(BaseModel):
    test_id:int
    @field_validator('test_id')
    @classmethod
    def validate_test_id(cls, test_id:int)->int:
        if 0<=test_id<=16669 : return test_id 
        else : raise ValueError('Test index out of range. Enter between 0-16669')

#=================================================================================
# Endpoint : /predict
#=================================================================================
'''This will recieve a input number between 0-16669 i.e the index of the test vector and
its corresponding SHAP values to send back to the frontend.'''

@app.post('/predict')
def predict(id:int)->JSONResponse:
    # Validation
    try:
        val_idx_obj=validate_idx(test_id=id)
    except ValidationError as e:
        raise HTTPException(
            status_code=422,
            detail=e.errors()[0]['msg']
        )
    
    test=val_idx_obj.test_id
    raw_row=session['vectors'].iloc[[test]]
    
    # Prob that txn is illicit
    prob=session['model'].predict(raw_row)[0]
    
    # Calculate the shap values
    shap_row=session['lookup'].iloc[test]
    raw_base = shap_row['Base Value']
    gnn_logits = shap_row['GNN Embeddings']
    local_logits = shap_row['Local Features']
    one_hop_logits = shap_row['One Hop Features']

    # Step-by-step accumulation in probability space
    base_prob = 1 / (1 + np.exp(-raw_base))
    prob_after_gnn = 1 / (1 + np.exp(-(raw_base + gnn_logits)))
    prob_after_local = 1 / (1 + np.exp(-(raw_base + gnn_logits + local_logits)))
    prob_after_one_hop = 1 / (1 + np.exp(-(raw_base + gnn_logits + local_logits + one_hop_logits)))

    # Isolate the exact shifts to send to the frontend
    gnn_shift = prob_after_gnn - base_prob
    local_shift = prob_after_local - prob_after_gnn
    one_hop_shift = prob_after_one_hop - prob_after_local

    return JSONResponse(status_code=200, content={
        'base_prob': float(base_prob),
        'gnn_shift': float(gnn_shift),
        'local_shift': float(local_shift),
        'one_hop_shift': float(one_hop_shift),
        'final_prob': float(prob)
    })

#=================================================================================
# Endpoint : /performance
#=================================================================================
'''This is the evidently AI performance check on the classification report. This takes random 5000 test vectors, makes prediction and prepares
the classification report using the ground truth labels.'''
@app.post('/performance')
def classification_report()->HTMLResponse:
    test_ids=np.random.choice(16669,5000,replace=False)
    
    sample_df=session['vectors'].iloc[test_ids].copy()
    predictions=session['model'].predict(sample_df)
    sample_df['prediction']=predictions
    sample_df['target']=session['truth'].iloc[test_ids].values

    definition = DataDefinition(
        classification=[
            BinaryClassification(
                target="target",
                prediction_labels="prediction"
            )
        ]
    )

    dataset = Dataset.from_pandas(
        sample_df,
        data_definition=definition
    )

    report = Report(metrics=[ClassificationPreset()])
    result=report.run(dataset)
    
    with NamedTemporaryFile(suffix=".html") as tmp:
        result.save_html(tmp.name)
        tmp.seek(0)
        html = tmp.read().decode("utf-8")

    return HTMLResponse(content=html)