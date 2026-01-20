const API_URL = "http://localhost:8000";

export interface PredictionResponse {
    label: "Normal" | "Pneumonia";
    confidence: number;
}

export async function checkHealth(): Promise<boolean> {
    try {
        const res = await fetch(`${API_URL}/health`);
        return res.ok;
    } catch {
        return false;
    }
}

export async function predictImage(file: File): Promise<PredictionResponse> {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
    });

    if (!res.ok) {
        const errorBody = await res.json().catch(() => ({}));
        throw new Error(errorBody.detail || "Failed to analyze image");
    }

    return res.json();
}
