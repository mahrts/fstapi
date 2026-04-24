## Usage:
This project can be runed locally as follow.

1- Clone the repo.

```bash
git clone <the repo>
```

2- Move to fstapi folder
```bash
cd fstapi
```

3- Install dependencies and the package:

```bash
pip install -r requirements.txt
pip install -e .
```

4- Tranin the model

```bash
python src/train_model.py
```
This will create a folder 'deployment', where necessary file deployment and testing are saved.

5- Finally, run the following and interact with full web interface 
(usually at http://127.0.0.1:8000). All usefull info to interact with the deployed model
can be found there.

```bash
uvicorn src.API.endpoints:app --reload
```