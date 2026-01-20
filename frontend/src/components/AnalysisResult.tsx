"use client";

import { CheckCircle, AlertTriangle } from "lucide-react";
import clsx from "clsx";

interface AnalysisResultProps {
    imageSrc: string | null;
    result: {
        label: "Normal" | "Pneumonia";
        confidence: number;
    } | null;
    onReset: () => void;
}

export default function AnalysisResult({ imageSrc, result, onReset }: AnalysisResultProps) {
    if (!imageSrc) {
        return (
            <div className="bg-clinical-white border border-clinical-border rounded-lg p-6 flex flex-col justify-center items-center text-clinical-text-sec min-h-[400px]">
                <p>No image analyzed</p>
            </div>
        );
    }

    const isPositive = result?.label === "Pneumonia";
    const confidencePercent = result ? Math.round(result.confidence * 100) : 0;

    return (
        <div className="bg-clinical-white border border-clinical-border rounded-lg overflow-hidden flex flex-col h-full min-h-[400px]">
            <div className="bg-gray-50 border-b border-clinical-border px-6 py-4 flex justify-between items-center">
                <h3 className="font-semibold text-clinical-text">Analysis Report</h3>
                {result && (
                    <span className={clsx(
                        "px-3 py-1 rounded-full text-xs font-medium flex items-center gap-1.5",
                        isPositive
                            ? "bg-red-50 text-red-700 border border-red-200"
                            : "bg-green-50 text-green-700 border border-green-200"
                    )}>
                        {isPositive ? <AlertTriangle size={14} /> : <CheckCircle size={14} />}
                        {result.label.toUpperCase()}
                    </span>
                )}
            </div>

            <div className="flex-1 p-6 flex flex-col gap-6">
                <div className="relative aspect-square w-full bg-black rounded-md overflow-hidden flex items-center justify-center border border-clinical-border">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={imageSrc} alt="Analyzed X-ray" className="object-contain w-full h-full" />
                </div>

                {result ? (
                    <div className="space-y-4">
                        <div>
                            <div className="flex justify-between text-sm mb-1">
                                <span className="font-medium text-clinical-text">Confidence Score</span>
                                <span className="text-clinical-text-sec">{confidencePercent}%</span>
                            </div>
                            <div className="w-full bg-gray-100 rounded-full h-2 overflow-hidden">
                                <div
                                    className={clsx("h-full rounded-full transition-all duration-1000", isPositive ? "bg-red-500" : "bg-medical-blue")}
                                    style={{ width: `${confidencePercent}%` }}
                                />
                            </div>
                        </div>

                        <div className="bg-blue-50 p-4 rounded-md border border-blue-100">
                            <p className="text-sm text-blue-800">
                                <strong>Clinical Note:</strong> The model has identified patterns consistent with {result.label.toLowerCase()} with {confidencePercent}% confidence. Please correlate with clinical findings.
                            </p>
                        </div>
                    </div>
                ) : (
                    <div className="flex-1 flex items-center justify-center">
                        <p className="text-clinical-text-sec animate-pulse">Processing...</p>
                    </div>
                )}
            </div>

            <div className="p-4 border-t border-clinical-border bg-gray-50 flex justify-end">
                <button
                    onClick={onReset}
                    className="text-sm font-medium text-clinical-text-sec hover:text-clinical-text px-4 py-2 hover:bg-gray-100 rounded-md transition-colors"
                >
                    Analyze New Image
                </button>
            </div>
        </div>
    );
}
