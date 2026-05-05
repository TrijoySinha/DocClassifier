import { useState } from "react";
import { predictImage } from "../api/predict";

export default function UploadBox() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const f = e.target.files[0];
    setFile(f);
    setPreview(URL.createObjectURL(f));
  };

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    setResult("");

    try {
      const res = await predictImage(file);
      setResult(res.prediction || res.label);
    } catch (err) {
      setResult("Error");
    }

    setLoading(false);
  };

  return (
    <div className="glass-card">
      <div className="title">Document Classifier</div>

      <label className="upload-box">
        Click to upload
        <input type="file" onChange={handleChange} />
      </label>

      {preview && <img src={preview} className="preview" />}

      <button
        className="button"
        onClick={handleUpload}
        disabled={!file || loading}
      >
        {loading ? <div className="loader"></div> : "Predict"}
      </button>

      {result && <div className="result">{result}</div>}
    </div>
  );
}