"use client";

import { useState } from "react";
import Head from "next/head";

export default function Home() {
  const [file, setFile] = useState(null);
  const [outputType, setOutputType] = useState("text");
  const [summary, setSummary] = useState("");
  const [audioUrl, setAudioUrl] = useState("");
  const [ttsAudioUrl, setTtsAudioUrl] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (e: any) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
  };

  const handleSubmit = async (e: any) => {
    e.preventDefault();
    if (!file) {
      setError("Please select a file first");
      return;
    }

    setIsLoading(true);
    setSummary("");
    setAudioUrl("");
    setTtsAudioUrl("");
    setError("");

    try {
      const formData = new FormData();
      const backendAPI = process.env.NEXT_PUBLIC_BACKEND_API || "";
      formData.append("file", file);
      formData.append("output_type", outputType);

      const response = await fetch(`${backendAPI}/api/summarize`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "An error occurred");
      }

      const data = await response.json();
      setSummary(data.summary);

      if (data.audio_url) {
        // Make sure this URL is accessible from your frontend
        setAudioUrl(`${backendAPI}${data.audio_url}`);
      }
      if (data.tts_audio_url) {
        setTtsAudioUrl(`${backendAPI}${data.tts_audio_url}`);
      }
    } catch (error: any) {
      setError(error.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8 max-w-4xl">
      <Head>
        <title>Document Summarizer</title>
        <meta name="description" content="Summarize documents with AI" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main>
        <h1 className="text-3xl font-bold mb-6 text-center">
          Document Summarizer
        </h1>

        <div className="bg-white rounded-lg text-black shadow-md p-6 mb-8">
          <form onSubmit={handleSubmit}>
            <div className="mb-4">
              <label className="block font-medium mb-2">
                Upload your document
              </label>
              <p className="text-sm mb-2">Supported formats: PDF, TXT</p>
              <input
                type="file"
                onChange={handleFileChange}
                accept=".pdf,.txt"
                className="w-full p-2 border border-gray-300 rounded"
              />
            </div>

            <div className="mb-4">
              <label className="block text-black font-medium mb-2">
                Output Type
              </label>
              <select
                value={outputType}
                onChange={(e) => setOutputType(e.target.value)}
                className="w-full p-2 border text-black border-gray-300 rounded"
              >
                <option value="text">Text Summary</option>
                <option value="audio">Audio Summary</option>
                <option value="notes">Notes</option>
                <option value="tts_audio">Audio Reading</option>
              </select>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:bg-blue-300"
            >
              {isLoading ? "Processing..." : "Upload & Summarize"}
            </button>
          </form>

          {isLoading && (
            <div className="flex justify-center my-6">
              <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-500"></div>
            </div>
          )}

          {error && (
            <div className="mt-4 p-3 bg-red-100 text-red-700 rounded">
              {error}
            </div>
          )}

          {summary && (
            <div className="mt-6">
              <h2 className="text-xl font-semibold mb-2">Summary</h2>
              <div className="p-4 bg-gray-50 rounded border text-black border-gray-200 whitespace-pre-wrap">
                {summary}
              </div>
            </div>
          )}

          {audioUrl && (
            <div className="mt-6">
              <h2 className="text-xl font-semibold mb-2">Audio Summary</h2>
              <audio controls className="w-full mt-2">
                <source src={audioUrl} type="audio/mpeg" />
                Your browser does not support the audio element.
              </audio>
              <div className="mt-2">
                <a
                  href={audioUrl}
                  download
                  className="text-blue-600 hover:underline"
                >
                  Download audio file
                </a>
              </div>
            </div>
          )}
          {ttsAudioUrl && (
            <div className="mt-6">
              <h2 className="text-xl font-semibold mb-2">Audio Summary</h2>
              <audio controls className="w-full mt-2">
                <source src={ttsAudioUrl} type="audio/mpeg" />
                Your browser does not support the audio element.
              </audio>
              <div className="mt-2">
                <a
                  href={ttsAudioUrl}
                  download
                  className="text-blue-600 hover:underline"
                >
                  Download audio file
                </a>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
