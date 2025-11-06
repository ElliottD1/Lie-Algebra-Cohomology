# Lie Algebra Cohomology

## Installation

### Prerequisites
- Python 3.7+
- pip

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running

1. Navigate to the `backend` directory:
```bash
cd backend
```

2. Start the server:
```bash
uvicorn app:app --reload
```

3. Open your browser to `http://localhost:8000`

## Usage

- **Group**: Select the Dynkin type (A, B, C, D, E, F, G)
- **n (rank)**: Rank of the Lie algebra
- **l (depth)**: Maximum depth for the Hasse diagram
- **Selected nodes**: Comma-separated list of vertices (1-based)
- **Use Adjoint Representation**: Use the adjoint representation's highest weight
- **Positive Gradation Only**: Show only positive gradations
