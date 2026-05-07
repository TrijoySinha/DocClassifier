import { useState } from "react";
import { predictImage } from "../api/predict";

export default function UploadBox() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState(null); 
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;

    setFile(selectedFile);
    setResults(null); // Clear previous results on new selection

    // Preview logic: only generate URL for images
    if (selectedFile.type.startsWith("image/")) {
      setPreview(URL.createObjectURL(selectedFile));
    } else {
      setPreview(null); // No preview for PDFs
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    setResults(null);

    try {
      // res expects: { type: "image", data: "label" } OR { type: "pdf", data: [...] }
      const res = await predictImage(file);
      setResults(res);
    } catch (err) {
      console.error("Upload Error:", err);
      setResults({ error: "Classification failed. Please try again." });
    } finally {
      setLoading(false);
    }
  };

  const clearSelection = () => {
    setFile(null);
    setPreview(null);
    setResults(null);
  };

  return (
    <div className="glass-card">
      <div className="title">Document Classifier</div>

      {!file ? (
        <label className="upload-box">
          Click to upload (IMG or PDF)
          <input 
            type="file" 
            accept=".jpg,.jpeg,.png,.pdf" 
            onChange={handleChange} 
            hidden 
          />
        </label>
      ) : (
        <div className="file-info-container">
            {preview ? (
                <img src={preview} className="preview" alt="Preview" />
            ) : (
                <div className="pdf-placeholder">
                    <span style={{ fontSize: "3rem" }}>📄</span>
                    <p>{file.name}</p>
                </div>
            )}
            <button className="clear-btn" onClick={clearSelection}>Change File</button>
        </div>
      )}

      <button 
        className="button" 
        onClick={handleUpload} 
        disabled={!file || loading}
      >
        {loading ? <div className="loader"></div> : "Predict Content"}
      </button>

      {/* --- Error Handling --- */}
      {results?.error && (
        <div className="result error">{results.error}</div>
      )}

      {/* --- Single Image Result --- */}
      {results?.type === "image" && (
        <div className="result">
            <span>Prediction:</span> <strong>{results.data}</strong>
        </div>
      )}

      {/* --- PDF Multi-page Results --- */}
      {results?.type === "pdf" && (
        <div className="pdf-results-container">
          <h3>PDF Analysis</h3>
          <div className="results-list">
            {results.data.map((item) => (
              <div key={item.page} className="page-row">
                <span>Page {item.page}</span>
                <span className="label-badge">{item.label}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}