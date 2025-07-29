import React from 'react';
import Container from '@mui/material/Container';
import Typography from '@mui/material/Typography';
import VideoFeed from './components/VideoFeed';

function App() {
  return (
    <Container maxWidth="md" style={{ marginTop: 32 }}>
      <Typography variant="h4" align="center" gutterBottom>
        OBEX Security Dashboard
      </Typography>
      <VideoFeed />
    </Container>
  );
}

export default App; 