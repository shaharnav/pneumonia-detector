"use client";

import { useState, useCallback } from "react";
import { UploadCloud, Loader2 } from "lucide-react";
import clsx from "clsx";

interface DropzoneProps {
    onFileSelect: (file: File) => void;
    isAnalyzing: boolean;
}

export default function Dropzone({ onFileSelect, isAnalyzing }: DropzoneProps) {
    const [isDragActive, setIsDragActive] = useState(false);

    const handleDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragActive(true);
    }, []);

    const handleDragLeave = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragActive(false);
    }, []);

    const handleDrop = useCallback(
        (e: React.DragEvent) => {
            e.preventDefault();
            setIsDragActive(false);

            if (isAnalyzing) return;

            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith("image/")) {
                onFileSelect(file);
            }
        },
        [onFileSelect, isAnalyzing]
    );

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            onFileSelect(e.target.files[0]);
        }
    };

    return (
        <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={clsx(
                "relative rounded-lg border-2 border-dashed p-10 transition-all duration-200 ease-in-out flex flex-col items-center justify-center text-center cursor-pointer min-h-[400px]",
                isDragActive
                    ? "border-medical-blue bg-blue-50/50"
                    : "border-clinical-border bg-clinical-white hover:bg-gray-50",
                isAnalyzing && "opacity-50 pointer-events-none"
            )}
        >
            <input
                type="file"
                accept="image/*"
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                onChange={handleChange}
                disabled={isAnalyzing}
            />

            <div className={clsx(
                "w-16 h-16 rounded-full flex items-center justify-center mb-4 transition-colors",
                isDragActive ? "bg-white text-medical-blue" : "bg-blue-50 text-medical-blue"
            )}>
                {isAnalyzing ? (
                    <Loader2 className="w-8 h-8 animate-spin" />
                ) : (
                    <UploadCloud className="w-8 h-8" />
                )}
            </div>

            <h3 className="text-lg font-medium text-clinical-text">
                {isAnalyzing ? "Analyzing X-ray..." : "Drag & drop X-ray image"}
            </h3>
            <p className="text-sm text-clinical-text-sec mt-2">
                {isAnalyzing ? "Please wait while our AI processes the image" : "or click to browse your files"}
            </p>

            {!isAnalyzing && (
                <div className="mt-6 text-xs text-clinical-text-sec/60">
                    Supports JPEG, PNG, DICOM (converted)
                </div>
            )}
        </div>
    );
}
