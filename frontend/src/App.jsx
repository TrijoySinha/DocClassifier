import './App.css';
import { useState } from 'react';

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setResult(null); // Reset result when a new file is picked
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) throw new Error("Server Error");
      
      const data = await response.json();
      setResult(data);
    } catch (error) {
      alert("Backend not reachable. Ensure main.py is running!");
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <div className="card">
        <h1 className="title">Document Classifier</h1>
        
        <div className="upload-section">
          <input 
            type="file" 
            className="file-input"
            accept="image/*" 
            onChange={handleFileChange} 
          />
          <button 
            className="upload-button"
            onClick={handleUpload} 
            disabled={loading || !file}
          >
            {loading ? 'Analyzing...' : 'Upload & Identify'}
          </button>
        </div>

        {result && (
          /* Dynamic class: if invalid, we can style it differently in CSS */
          <div className={`result-container ${result.label === "Invalid Document" ? 'invalid' : 'valid'}`}>
            <div className="result-item">
              <span className="res-tag">Status</span>
              <div className="res-value">{result.label}</div>
            </div>
            <div className="result-item">
              <span className="res-tag">Confidence</span>
              <div className="res-conf">{result.confidence}</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
