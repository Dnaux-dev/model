import React, { useEffect, useRef, useState } from 'react';
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Switch from '@mui/material/Switch';
import FormControlLabel from '@mui/material/FormControlLabel';

const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 480;

function VideoFeed() {
  const [alerts, setAlerts] = useState([]);
  const [latestAlert, setLatestAlert] = useState(null);
  const [zone, setZone] = useState(null); // {x1, y1, x2, y2}
  const [drawing, setDrawing] = useState(false);
  const [startPoint, setStartPoint] = useState(null);
  const [endPoint, setEndPoint] = useState(null);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [heatmapUrl, setHeatmapUrl] = useState('');
  const videoRef = useRef();
  const canvasRef = useRef();
  const heatmapInterval = useRef();

  // Poll alerts
  useEffect(() => {
    const fetchAlerts = async () => {
      const loiterRes = await fetch('http://localhost:8000/loitering_alerts');
      const intrusionRes = await fetch('http://localhost:8000/intrusion_alerts');
      const loiterData = await loiterRes.json();
      const intrusionData = await intrusionRes.json();
      let allAlerts = [];
      if (loiterData.loitering_alerts) {
        allAlerts = allAlerts.concat(
          loiterData.loitering_alerts.map(a => ({
            type: 'loitering',
            id: a.track_id,
            time: new Date(a.entry_time * 1000).toLocaleTimeString(),
            duration: a.duration ? a.duration.toFixed(1) : null
          }))
        );
      }
      if (intrusionData.intrusion_alerts) {
        allAlerts = allAlerts.concat(
          intrusionData.intrusion_alerts.map(a => ({
            type: 'intrusion',
            id: a.track_id,
            time: new Date(a.entry_time * 1000).toLocaleTimeString()
          }))
        );
      }
      // Sort by time descending
      allAlerts.sort((a, b) => b.time.localeCompare(a.time));
      setAlerts(allAlerts);
      setLatestAlert(allAlerts[0] || null);
    };
    fetchAlerts();
    const interval = setInterval(fetchAlerts, 1000);
    return () => clearInterval(interval);
  }, []);

  // Draw overlays (zone, drawing rectangle)
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Draw zone
    if (zone) {
      ctx.strokeStyle = 'red';
      ctx.lineWidth = 3;
      ctx.strokeRect(zone.x1, zone.y1, zone.x2 - zone.x1, zone.y2 - zone.y1);
    }
    // Draw current drawing rectangle
    if (drawing && startPoint && endPoint) {
      ctx.strokeStyle = 'blue';
      ctx.lineWidth = 2;
      const x = Math.min(startPoint.x, endPoint.x);
      const y = Math.min(startPoint.y, endPoint.y);
      const w = Math.abs(startPoint.x - endPoint.x);
      const h = Math.abs(startPoint.y - endPoint.y);
      ctx.strokeRect(x, y, w, h);
    }
  }, [zone, drawing, startPoint, endPoint]);

  // Heatmap polling
  useEffect(() => {
    if (showHeatmap) {
      const fetchHeatmap = () => {
        // Add a cache buster to always get the latest image
        setHeatmapUrl(`http://localhost:8000/heatmap?${Date.now()}`);
      };
      fetchHeatmap();
      heatmapInterval.current = setInterval(fetchHeatmap, 1000);
      return () => clearInterval(heatmapInterval.current);
    } else {
      setHeatmapUrl('');
      if (heatmapInterval.current) clearInterval(heatmapInterval.current);
    }
  }, [showHeatmap]);

  // Mouse event handlers for drawing zone
  const handleCanvasMouseDown = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const x = Math.round((e.clientX - rect.left));
    const y = Math.round((e.clientY - rect.top));
    setDrawing(true);
    setStartPoint({ x, y });
    setEndPoint({ x, y });
  };

  const handleCanvasMouseMove = (e) => {
    if (!drawing) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = Math.round((e.clientX - rect.left));
    const y = Math.round((e.clientY - rect.top));
    setEndPoint({ x, y });
  };

  const handleCanvasMouseUp = async (e) => {
    if (!drawing) return;
    setDrawing(false);
    const rect = canvasRef.current.getBoundingClientRect();
    const x2 = Math.round((e.clientX - rect.left));
    const y2 = Math.round((e.clientY - rect.top));
    const x1 = startPoint.x;
    const y1 = startPoint.y;
    // Normalize coordinates
    const zoneCoords = {
      x1: Math.min(x1, x2),
      y1: Math.min(y1, y2),
      x2: Math.max(x1, x2),
      y2: Math.max(y1, y2)
    };
    setZone(zoneCoords);
    // Send to backend
    await fetch('http://localhost:8000/set_zone', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(zoneCoords)
    });
  };

  // Download heatmap handler
  const handleDownloadHeatmap = () => {
    if (!heatmapUrl) return;
    const link = document.createElement('a');
    link.href = heatmapUrl;
    link.download = `heatmap_${Date.now()}.jpg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <Box position="relative" width={VIDEO_WIDTH} height={VIDEO_HEIGHT}>
      {/* Video feed */}
      <img
        ref={videoRef}
        src="http://localhost:8000/video_feed"
        alt="Video Feed"
        width={VIDEO_WIDTH}
        height={VIDEO_HEIGHT}
        style={{ display: 'block' }}
      />
      {/* Heatmap overlay */}
      {showHeatmap && heatmapUrl && (
        <img
          src={heatmapUrl}
          alt="Heatmap Overlay"
          width={VIDEO_WIDTH}
          height={VIDEO_HEIGHT}
          style={{
            position: 'absolute',
            left: 0,
            top: 0,
            opacity: 0.5,
            pointerEvents: 'none',
            zIndex: 4,
            mixBlendMode: 'screen',
          }}
        />
      )}
      {/* Canvas overlay for zone and drawing */}
      <canvas
        ref={canvasRef}
        width={VIDEO_WIDTH}
        height={VIDEO_HEIGHT}
        style={{ position: 'absolute', left: 0, top: 0, pointerEvents: 'auto', zIndex: 2 }}
        onMouseDown={handleCanvasMouseDown}
        onMouseMove={handleCanvasMouseMove}
        onMouseUp={handleCanvasMouseUp}
      />
      {/* Draw Zone Button */}
      <Button
        variant="contained"
        color="primary"
        sx={{ position: 'absolute', top: 16, right: 16, zIndex: 5 }}
        onClick={() => { setZone(null); setStartPoint(null); setEndPoint(null); }}
      >
        Draw Zone
      </Button>
      {/* Heatmap Toggle and Download */}
      <Box sx={{ position: 'absolute', top: 16, left: 16, zIndex: 5, display: 'flex', flexDirection: 'column', gap: 1 }}>
        <FormControlLabel
          control={<Switch checked={showHeatmap} onChange={e => setShowHeatmap(e.target.checked)} color="secondary" />}
          label="Show Heatmap"
        />
        <Button
          variant="outlined"
          color="secondary"
          onClick={handleDownloadHeatmap}
          disabled={!heatmapUrl}
        >
          Download Heatmap
        </Button>
      </Box>
      {/* Latest alert overlay */}
      {latestAlert && (
        <Paper
          elevation={6}
          sx={{
            position: 'absolute',
            top: 24,
            left: '50%',
            transform: 'translateX(-50%)',
            bgcolor: latestAlert.type === 'intrusion' ? 'error.main' : 'warning.main',
            color: 'white',
            px: 4,
            py: 2,
            fontSize: 24,
            zIndex: 10,
            minWidth: 320,
            textAlign: 'center',
          }}
        >
          <Typography variant="h5" fontWeight="bold">
            {latestAlert.type === 'intrusion'
              ? `ZONE INTRUSION ALERT!`
              : `LOITERING DETECTED!`}
          </Typography>
          <Typography>
            Person ID: {latestAlert.id}
            {latestAlert.type === 'loitering' && latestAlert.duration
              ? ` (Duration: ${latestAlert.duration}s)`
              : ''}
          </Typography>
          <Typography variant="caption">{latestAlert.time}</Typography>
        </Paper>
      )}
      {/* Alert log at bottom */}
      <Box
        sx={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          width: '100%',
          bgcolor: 'rgba(0,0,0,0.7)',
          color: 'white',
          px: 2,
          py: 1,
          maxHeight: 120,
          overflowY: 'auto',
        }}
      >
        {alerts.map((alert, idx) => (
          <Typography key={idx} fontSize={14}>
            [{alert.time}] {alert.type === 'intrusion' ? 'Zone intrusion' : 'Loitering'} - Person ID: {alert.id}
            {alert.type === 'loitering' && alert.duration ? ` (Duration: ${alert.duration}s)` : ''}
          </Typography>
        ))}
      </Box>
    </Box>
  );
}

export default VideoFeed; 