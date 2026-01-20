"use client";

import { useState } from "react";
import Dropzone from "@/components/Dropzone";
import AnalysisResult from "@/components/AnalysisResult";
import { predictImage, PredictionResponse } from "@/lib/api";

export default function Home() {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelect = async (selectedFile: File) => {
    setPreviewUrl(URL.createObjectURL(selectedFile));
    setIsAnalyzing(true);
    setError(null);
    setResult(null);

    try {
      const data = await predictImage(selectedFile);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setError(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
  };

  return (
    <main className="min-h-screen bg-clinical-bg text-clinical-text pb-10">
      {/* Header - Simple, clean, minimal */}
      <header className="bg-clinical-white border-b border-clinical-border h-16 flex items-center px-8 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-medical-blue rounded-md flex items-center justify-center text-white">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="w-5 h-5"
            >
              <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
            </svg>
          </div>
          <h1 className="font-semibold text-lg tracking-tight">
            Pneumonia<span className="text-medical-blue">Detector</span>
          </h1>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-5xl mx-auto mt-10 px-6">
        <div className="mb-8">
          <h2 className="text-2xl font-semibold mb-2">Analysis Dashboard</h2>
          <p className="text-clinical-text-sec">
            Upload chest X-ray images for rapid pneumonia screening.
          </p>
        </div>

        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
            Error: {error}
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Upload Zone */}
          <Dropzone onFileSelect={handleFileSelect} isAnalyzing={isAnalyzing} />

          {/* Results Zone */}
          <AnalysisResult imageSrc={previewUrl} result={result} onReset={handleReset} />
        </div>
      </div>
    </main>
  );
}
