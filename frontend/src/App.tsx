import React, { useCallback, useMemo, useState } from "react";

type PredictionLabel = "Normal" | "Pneumonia";

interface PredictionResult {
  label: PredictionLabel;
  confidence: number;
}

const API_URL = "http://localhost:8000/predict";

const App: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileUploaded, setFileUploaded] = useState(false);

  const handleFiles = useCallback((files: FileList | null) => {
    if (!files || files.length === 0) return;

    const candidate = files[0];
    if (!candidate.type.startsWith("image/")) {
      setError("Please upload a valid image file (JPEG, PNG, etc.).");
      setFile(null);
      setPreviewUrl(null);
      setPrediction(null);
      return;
    }

    setError(null);
    setFile(candidate);
    setPrediction(null);
    setFileUploaded(true);

    const reader = new FileReader();
    reader.onload = () => {
      setPreviewUrl(reader.result as string);
    };
    reader.readAsDataURL(candidate);
  }, []);

  const onDrop: React.DragEventHandler<HTMLDivElement> = (event) => {
    event.preventDefault();
    event.stopPropagation();
    handleFiles(event.dataTransfer.files);
  };

  const onDragOver: React.DragEventHandler<HTMLDivElement> = (event) => {
    event.preventDefault();
  };

  const onFileChange: React.ChangeEventHandler<HTMLInputElement> = (event) => {
    handleFiles(event.target.files);
  };

  const analyze = async () => {
    if (!file) {
      setError("Please upload an X-ray image before analyzing.");
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const data = await response.json().catch(() => null);
        const message =
          data?.detail ??
          `Analysis failed with status ${response.status}. Please try again.`;
        throw new Error(message);
      }

      const data = (await response.json()) as PredictionResult;
      setPrediction(data);
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Unexpected error during analysis. Please try again.";
      setError(message);
      setPrediction(null);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const resetUpload = () => {
    setFile(null);
    setPreviewUrl(null);
    setPrediction(null);
    setFileUploaded(false);
    setError(null);
  };

  const confidencePercent = useMemo(() => {
    if (!prediction) return 0;
    const pct = Math.round(prediction.confidence * 100);
    return Math.min(Math.max(pct, 0), 100);
  }, [prediction]);

  return (
    <div className="min-h-screen bg-slate-950">
      <div className="mx-auto flex max-w-6xl flex-col gap-8 px-4 py-10 lg:flex-row">
        <section className="flex-1 space-y-6">
          <header className="space-y-2">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-sky-400">
              Radiology AI Assistant
            </p>
            <h1 className="text-3xl font-semibold tracking-tight text-slate-50 sm:text-4xl">
              Pneumonia Detection
            </h1>
            <p className="max-w-xl text-sm text-slate-400 sm:text-base">
              Upload a chest X-ray to obtain an AI-assisted assessment for
              pneumonia. This tool is intended for research use and does not
              replace clinical judgment.
            </p>
          </header>

          {!fileUploaded ? (
            <div
              onDrop={onDrop}
              onDragOver={onDragOver}
              className="group relative flex cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-slate-600 bg-slate-800/50 px-6 py-10 text-center transition hover:border-cyan-500 hover:bg-slate-800"
            >
              <input
                type="file"
                accept="image/*"
                onChange={onFileChange}
                className="absolute inset-0 z-10 h-full w-full cursor-pointer opacity-0"
              />
              <div className="pointer-events-none space-y-3">
                <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-slate-700 text-cyan-400 ring-1 ring-slate-600">
                  <span className="text-xl">⬆</span>
                </div>
                <div className="space-y-1">
                  <p className="text-sm font-medium text-slate-100">
                    Upload chest X-ray
                  </p>
                  <p className="text-xs text-slate-400">
                    Drag and drop a DICOM-exported JPEG/PNG or click to browse.
                  </p>
                </div>
                <p className="text-[11px] text-slate-500">
                  Recommended: PA/AP view · Up to 10 MB
                </p>
              </div>
            </div>
          ) : (
            <div className="rounded-xl border border-cyan-500/30 bg-slate-800/60 px-6 py-6">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-cyan-500/20 text-cyan-400 ring-2 ring-cyan-500/30">
                  <svg
                    className="h-5 w-5"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M5 13l4 4L19 7"
                    />
                  </svg>
                </div>
                <div className="flex-1">
                  <p className="text-sm font-medium text-slate-100">
                    File Loaded: <span className="text-cyan-300">{file?.name}</span>
                  </p>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Ready for analysis
                  </p>
                </div>
                <button
                  type="button"
                  onClick={resetUpload}
                  className="rounded-lg border border-slate-600 bg-slate-700 px-4 py-2 text-xs font-medium text-slate-200 transition hover:bg-slate-600 hover:text-slate-100"
                >
                  Upload another X-ray
                </button>
              </div>
            </div>
          )}

          {error && (
            <div className="rounded-lg border border-red-500/40 bg-red-950/40 px-4 py-3 text-sm text-red-100">
              {error}
            </div>
          )}

          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={analyze}
              disabled={isAnalyzing || !fileUploaded}
              className="inline-flex items-center justify-center gap-2 rounded-lg bg-cyan-600 px-6 py-2.5 text-sm font-medium text-white shadow-sm shadow-cyan-900/40 transition hover:bg-cyan-500 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:shadow-none"
            >
              {isAnalyzing ? (
                <>
                  <svg
                    className="h-4 w-4 animate-spin"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    ></circle>
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    ></path>
                  </svg>
                  Analyzing...
                </>
              ) : (
                "Analyze"
              )}
            </button>
          </div>
        </section>

        <section className="flex-1 space-y-4">
          <div className="rounded-xl border border-slate-700 bg-slate-800/60 p-5 backdrop-blur">
            <h2 className="text-sm font-semibold uppercase tracking-[0.16em] text-slate-400">
              Clinical Report
            </h2>

            {!prediction && !previewUrl && (
              <p className="mt-4 text-sm text-slate-500">
                Upload an X-ray and start the analysis to view an AI-assisted
                report.
              </p>
            )}

            {previewUrl && (
              <div className="mt-4 flex gap-4">
                <div className="w-32 flex-shrink-0 overflow-hidden rounded-lg border border-slate-700 bg-black/60">
                  <img
                    src={previewUrl}
                    alt="Uploaded chest X-ray"
                    className="h-full w-full object-cover"
                  />
                </div>

                <div className="flex-1 space-y-4">
                  {prediction ? (
                    <>
                      <div className="flex items-baseline justify-between gap-4">
                        <p className="text-sm font-medium text-slate-300">
                          Impression
                        </p>
                        <span
                          className={`inline-flex items-center rounded-lg px-3 py-0.5 text-xs font-semibold ${
                            prediction.label === "Pneumonia"
                              ? "bg-red-500/10 text-red-300 ring-1 ring-red-500/40"
                              : "bg-cyan-500/10 text-cyan-300 ring-1 ring-cyan-500/40"
                          }`}
                        >
                          {prediction.label}
                        </span>
                      </div>

                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-xs text-slate-400">
                          <span>Model confidence</span>
                          <span className="font-medium text-slate-100">
                            {confidencePercent}%
                          </span>
                        </div>
                        <div className="h-2 overflow-hidden rounded-full bg-slate-700">
                          <div
                            className={`h-full rounded-full transition-all ${
                              prediction.label === "Pneumonia"
                                ? "bg-red-500"
                                : "bg-cyan-500"
                            }`}
                            style={{ width: `${confidencePercent}%` }}
                          />
                        </div>
                      </div>

                      <div className="space-y-1 text-xs text-slate-400">
                        <p className="font-medium text-slate-300">
                          Interpretation
                        </p>
                        <p>
                          This automated assessment analyzes pixel-level
                          patterns consistent with{" "}
                          <span className="font-semibold text-slate-100">
                            {prediction.label.toLowerCase()}
                          </span>
                          . Always correlate with clinical findings and
                          radiologist review.
                        </p>
                      </div>
                    </>
                  ) : (
                    <p className="text-sm text-slate-500">
                      X-ray loaded. Run analysis to generate an AI summary.
                    </p>
                  )}
                </div>
              </div>
            )}
          </div>

          <p className="text-[11px] text-slate-500">
            This prototype is not approved for clinical use. Do not use it for
            diagnosis or treatment decisions.
          </p>
        </section>
      </div>
    </div>
  );
};

export default App;

