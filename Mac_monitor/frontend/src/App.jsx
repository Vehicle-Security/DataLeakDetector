import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Home from './pages/Home';
import SessionDetail from './pages/SessionDetail';
import './index.css';

function App() {
  return (
    <BrowserRouter>
      <div className="bg-[#0f111a] min-h-screen text-gray-300 font-sans">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/session/:id" element={<SessionDetail />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;
