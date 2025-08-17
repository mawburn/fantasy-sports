import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Home } from "@/pages/Home";
import { Upload } from "@/pages/Upload";
import { Optimize } from "@/pages/Optimize";
import { Predictions } from "@/pages/Predictions";

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-background">
        <nav className="border-b border-border">
          <div className="container mx-auto px-4 py-4">
            <div className="flex items-center justify-between">
              <Link to="/" className="text-2xl font-bold">
                NFL DFS
              </Link>
              <div className="flex space-x-4">
                <Button variant="ghost" asChild>
                  <Link to="/">Home</Link>
                </Button>
                <Button variant="ghost" asChild>
                  <Link to="/upload">Upload</Link>
                </Button>
                <Button variant="ghost" asChild>
                  <Link to="/optimize">Optimize</Link>
                </Button>
                <Button variant="ghost" asChild>
                  <Link to="/predictions">Predictions</Link>
                </Button>
              </div>
            </div>
          </div>
        </nav>

        <main>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/upload" element={<Upload />} />
            <Route path="/optimize" element={<Optimize />} />
            <Route path="/predictions" element={<Predictions />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
