import React, { useRef, useEffect, useState, useCallback } from 'react';
import { PenTool, X, Save, Undo, Trash2 } from 'lucide-react';
import { Button } from './ui/button';

export function VideoFeed({ isDrawing, setIsDrawing, onSaveFence, fences, onDeleteFence }) {
  const canvasRef = useRef(null);
  const videoWrapperRef = useRef(null);
  const [points, setPoints] = useState([]);
  
  // Handle resize and redrawing
  useEffect(() => {
    const handleResize = () => {
      if (canvasRef.current && videoWrapperRef.current) {
        const rect = videoWrapperRef.current.getBoundingClientRect();
        canvasRef.current.width = rect.width;
        canvasRef.current.height = rect.height;
        drawCanvas();
      }
    };
    
    window.addEventListener('resize', handleResize);
    // Initial size
    setTimeout(handleResize, 100);
    
    return () => window.removeEventListener('resize', handleResize);
  }, [points, isDrawing]);

  // Undo (Ctrl+Z) logic
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.ctrlKey && e.key === 'z' && isDrawing) {
        setPoints(prev => prev.slice(0, -1));
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isDrawing]);

  // Clear points when drawing mode is toggled on externally
  useEffect(() => {
    if (isDrawing) {
      setPoints([]);
    }
  }, [isDrawing]);

  // Calculate actual rendered image metrics to fix normalized coordinates
  const getImageMetrics = useCallback(() => {
    const canvas = canvasRef.current;
    const img = document.getElementById('cctv-feed');
    if (!canvas || !img) return null;
    
    const containerRatio = canvas.width / canvas.height;
    // use natural sizes, fallback if not loaded
    const naturalWidth = img.naturalWidth || 640;
    const naturalHeight = img.naturalHeight || 480;
    const imageRatio = naturalWidth / naturalHeight;
    
    let renderWidth, renderHeight, offsetX = 0, offsetY = 0;
    
    if (containerRatio > imageRatio) {
      renderHeight = canvas.height;
      renderWidth = canvas.height * imageRatio;
      offsetX = (canvas.width - renderWidth) / 2;
    } else {
      renderWidth = canvas.width;
      renderHeight = canvas.width / imageRatio;
      offsetY = (canvas.height - renderHeight) / 2;
    }
    
    return { renderWidth, renderHeight, offsetX, offsetY };
  }, []);

  // Canvas Drawing logic
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (!isDrawing || points.length === 0) return;

    const metrics = getImageMetrics();
    if (!metrics) return;
    const { renderWidth, renderHeight, offsetX, offsetY } = metrics;

    ctx.strokeStyle = '#00f0ff';
    ctx.fillStyle = 'rgba(0, 240, 255, 0.25)';
    ctx.lineWidth = 2;

    ctx.beginPath();
    points.forEach((pt, idx) => {
      const x = pt.normX * renderWidth + offsetX;
      const y = pt.normY * renderHeight + offsetY;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);

      // Draw vertex handle
      ctx.fillStyle = '#ff0055';
      ctx.fillRect(x - 4, y - 4, 8, 8);
      ctx.fillStyle = 'rgba(0, 240, 255, 0.25)';
    });

    if (points.length > 2) {
      ctx.closePath();
      ctx.fill();
    }
    ctx.stroke();
  }, [isDrawing, points, getImageMetrics]);

  useEffect(() => {
    drawCanvas();
  }, [drawCanvas]);

  const handleCanvasClick = (e) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;

    const metrics = getImageMetrics();
    if (!metrics) return;
    const { renderWidth, renderHeight, offsetX, offsetY } = metrics;

    // Reject clicks outside the letterboxed video area
    if (px < offsetX || px > offsetX + renderWidth || py < offsetY || py > offsetY + renderHeight) {
      return;
    }

    const normX = Math.max(0, Math.min(1, (px - offsetX) / renderWidth));
    const normY = Math.max(0, Math.min(1, (py - offsetY) / renderHeight));
    
    setPoints([...points, { normX, normY }]);
  };

  const handleCancelDraw = () => {
    setIsDrawing(false);
    setPoints([]);
  };

  const handleUndo = () => {
    setPoints(prev => prev.slice(0, -1));
  };

  const handleSave = () => {
    if (points.length < 3) {
      alert("Virtual fence requires at least 3 vertices to enclose a zone!");
      return;
    }
    onSaveFence(points);
  };

  return (
    <div className="relative flex-1 flex flex-col bg-black overflow-hidden rounded-xl border border-zinc-800 shadow-lg">
      {isDrawing && (
        <div className="absolute top-4 right-4 z-50 flex gap-2">
          <Button onClick={handleCancelDraw} variant="destructive">
            <X className="mr-2 h-4 w-4" /> CANCEL
          </Button>
          <Button onClick={handleUndo} variant="outline" className="bg-background/80 hover:bg-background">
            <Undo className="mr-2 h-4 w-4" /> UNDO
          </Button>
          <Button onClick={handleSave} variant="default" className="bg-green-600 hover:bg-green-700 text-white">
            <Save className="mr-2 h-4 w-4" /> SAVE ZONE
          </Button>
        </div>
      )}

      {isDrawing && (
        <div className="absolute top-4 left-4 z-50 bg-black/60 text-primary border border-primary/50 px-4 py-2 rounded-md font-mono text-sm pointer-events-none">
          DRAW MODE ACTIVE: Click on the feed to place vertices. (Ctrl+Z to Undo)
        </div>
      )}



      <div ref={videoWrapperRef} className="relative w-full h-full flex items-center justify-center bg-zinc-950">
        {/* We use the native img tag fetching MJPEG feed */}
        <img 
          id="cctv-feed"
          src="/video_feed" 
          alt="Live Video Feed"
          className="w-full h-full object-contain pointer-events-none"
          onLoad={drawCanvas}
          onError={(e) => {
             setTimeout(() => {
               e.target.src = '/video_feed?t=' + Date.now();
             }, 2500);
          }}
        />
        
        <canvas
          ref={canvasRef}
          onClick={handleCanvasClick}
          className={`absolute top-0 left-0 w-full h-full ${isDrawing ? 'cursor-crosshair z-40 pointer-events-auto' : 'pointer-events-none z-10'}`}
        />
      </div>
    </div>
  );
}
