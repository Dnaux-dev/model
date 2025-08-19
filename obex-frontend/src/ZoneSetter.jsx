// Example: obex-frontend/src/components/ZoneSetter.jsx
import React, { useState } from 'react';
import { setZone } from '../services/backendIntegration';

export default function ZoneSetter() {
  const [zone, setZoneState] = useState({ x1: '', y1: '', x2: '', y2: '' });
  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    setZoneState({ ...zone, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const res = await setZone(
      parseInt(zone.x1),
      parseInt(zone.y1),
      parseInt(zone.x2),
      parseInt(zone.y2)
    );
    setResult(res);
  };

  return (
    <div>
      <h3>Set Monitoring Zone</h3>
      <form onSubmit={handleSubmit}>
        <input name="x1" placeholder="x1" value={zone.x1} onChange={handleChange} required />
        <input name="y1" placeholder="y1" value={zone.y1} onChange={handleChange} required />
        <input name="x2" placeholder="x2" value={zone.x2} onChange={handleChange} required />
        <input name="y2" placeholder="y2" value={zone.y2} onChange={handleChange} required />
        <button type="submit">Set Zone</button>
      </form>
      {result && <pre>{JSON.stringify(result, null, 2)}</pre>}
    </div>
  );
}