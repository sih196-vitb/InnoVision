import React, { useState, useEffect, useRef } from 'react';
import { io } from 'socket.io-client';
import { Header } from './components/Header';
import { VideoFeed } from './components/VideoFeed';
import { AlertFeed } from './components/AlertFeed';
import { TelemetryBar } from './components/TelemetryBar';
import { FenceModal } from './components/FenceModal';
import { ActiveFencesList } from './components/ActiveFencesList';

export default function App() {
  const [stats, setStats] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [fences, setFences] = useState([]);
  
  const [isDrawing, setIsDrawing] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [currentFencePoints, setCurrentFencePoints] = useState([]);
  const [audioMuted, setAudioMuted] = useState(false);

  const audioCtxRef = useRef(null);

  // Sound synthesis
  const playAlertChime = (threatLevel) => {
    if (audioMuted) return;
    try {
      if (!audioCtxRef.current) {
        audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)();
      }
      const audioCtx = audioCtxRef.current;
      if (audioCtx.state === 'suspended') {
        audioCtx.resume();
      }

      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.connect(gain);
      gain.connect(audioCtx.destination);

      const now = audioCtx.currentTime;
      if (threatLevel === 'CRITICAL') {
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(880, now);
        osc.frequency.setValueAtTime(1100, now + 0.1);
        gain.gain.setValueAtTime(0.3, now);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 0.35);
        osc.start(now);
        osc.stop(now + 0.35);
      } else {
        osc.type = 'sine';
        osc.frequency.setValueAtTime(640, now);
        gain.gain.setValueAtTime(0.15, now);
        gain.gain.exponentialRampToValueAtTime(0.01, now + 0.2);
        osc.start(now);
        osc.stop(now + 0.2);
      }
    } catch (e) {
      console.warn("Audio synthesis:", e);
    }
  };

  // Socket.IO and Polling
  useEffect(() => {
    let socket;
    try {
      socket = io();
      socket.on('connect', () => console.log('[WebSocket] Connected.'));
      socket.on('alert_event', (alert) => {
        setAlerts(prev => [alert, ...prev].slice(0, 50));
        playAlertChime(alert.threat_level);
      });
      socket.on('fences_updated', () => loadFences());
    } catch (e) {
      console.warn('[WebSocket] Init error:', e);
    }

    const pollInterval = setInterval(async () => {
      try {
        const statsRes = await fetch('/api/stats');
        const statsData = await statsRes.json();
        setStats(statsData);
      } catch (e) {}
    }, 2000);

    return () => {
      if (socket) socket.disconnect();
      clearInterval(pollInterval);
    };
  }, [audioMuted]);

  const loadFences = async () => {
    try {
      const res = await fetch('/api/fences');
      const data = await res.json();
      setFences(data.fences || []);
    } catch (e) {}
  };

  useEffect(() => {
    loadFences();
  }, []);

  const handleSaveFenceAttempt = (points) => {
    setCurrentFencePoints(points);
    setIsModalOpen(true);
  };

  const handleFenceModalSave = async (fenceConfig) => {
    const polygon = currentFencePoints.map(pt => [
      Number(Math.round(pt.normX + 'e4') + 'e-4'),
      Number(Math.round(pt.normY + 'e4') + 'e-4')
    ]);

    const newFence = {
      id: `fence_${Date.now()}`,
      name: fenceConfig.name,
      zone_type: fenceConfig.zoneType,
      color: fenceConfig.zoneType === 'RESTRICTED_PERIMETER' ? '#EF4444' : '#F59E0B',
      polygon: polygon,
      allowed_classes: fenceConfig.zoneType === 'RESTRICTED_PERIMETER' ? [] : [0, 2, 7],
      loiter_threshold_sec: fenceConfig.loiter
    };

    try {
      const updatedFences = [...fences, newFence];
      await fetch('/api/fences', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fences: updatedFences })
      });
      
      setFences(updatedFences);
      setIsModalOpen(false);
      setIsDrawing(false);
      setCurrentFencePoints([]);
    } catch (e) {
      alert("Error saving virtual fence: " + e);
    }
  };

  const handleDeleteFence = async (id) => {
    try {
      const updatedFences = fences.filter(f => f.id !== id);
      await fetch('/api/fences', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fences: updatedFences })
      });
      setFences(updatedFences);
    } catch (e) {
      alert("Error deleting fence: " + e);
    }
  };

  return (
    <div className="flex h-screen flex-col bg-background text-foreground overflow-hidden">
      <Header 
        stats={stats} 
        audioMuted={audioMuted} 
        toggleAudio={() => setAudioMuted(!audioMuted)} 
        isDrawing={isDrawing}
        setIsDrawing={setIsDrawing}
      />
      
      <div className="flex flex-1 overflow-hidden pt-2 pb-4 px-4 gap-4 bg-zinc-950/50">
        <VideoFeed 
          isDrawing={isDrawing} 
          setIsDrawing={setIsDrawing} 
          onSaveFence={handleSaveFenceAttempt} 
          fences={fences}
          onDeleteFence={handleDeleteFence}
        />
        <div className="w-[300px] flex-shrink-0 hidden md:flex md:flex-col gap-4">
          <ActiveFencesList fences={fences} onDeleteFence={handleDeleteFence} />
          <AlertFeed alerts={alerts} onClear={() => setAlerts([])} />
        </div>
      </div>
      
      <TelemetryBar stats={stats} />

      <FenceModal 
        isOpen={isModalOpen} 
        onClose={() => setIsModalOpen(false)} 
        onSave={handleFenceModalSave} 
      />
    </div>
  );
}
