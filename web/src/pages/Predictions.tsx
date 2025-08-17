import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useEffect, useState } from "react";

interface PlayerPrediction {
  name: string;
  position: string;
  team: string;
  projected_points: number;
  confidence: number;
  salary?: number;
}

export function Predictions() {
  const [predictions, setPredictions] = useState<PlayerPrediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    fetchPredictions();
  }, []);

  const fetchPredictions = async () => {
    try {
      const response = await fetch(
        "http://localhost:8000/api/predictions/players",
      );
      if (response.ok) {
        const data = await response.json();
        setPredictions(data);
      }
    } catch (error) {
      console.error("Failed to fetch predictions:", error);
    } finally {
      setLoading(false);
    }
  };

  const filteredPredictions = predictions.filter(
    (player) =>
      player.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      player.position.toLowerCase().includes(searchTerm.toLowerCase()) ||
      player.team.toLowerCase().includes(searchTerm.toLowerCase()),
  );

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="text-center">Loading predictions...</div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">Player Predictions</h1>

        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Search Players</CardTitle>
          </CardHeader>
          <CardContent>
            <Input
              placeholder="Search by name, position, or team..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </CardContent>
        </Card>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredPredictions.map((player, index) => (
            <Card key={index}>
              <CardHeader>
                <CardTitle className="text-lg">{player.name}</CardTitle>
                <CardDescription>
                  {player.position} - {player.team}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Projected Points:</span>
                    <span className="font-medium">
                      {player.projected_points.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Confidence:</span>
                    <span className="font-medium">
                      {(player.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  {player.salary && (
                    <div className="flex justify-between">
                      <span>Salary:</span>
                      <span className="font-medium">
                        ${player.salary.toLocaleString()}
                      </span>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {filteredPredictions.length === 0 && (
          <div className="text-center py-8 text-muted-foreground">
            No predictions found matching your search.
          </div>
        )}
      </div>
    </div>
  );
}
