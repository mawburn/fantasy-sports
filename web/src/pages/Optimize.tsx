import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useState } from "react";

interface LineupPlayer {
  name: string;
  position: string;
  salary: number;
  projected_points: number;
}

interface Lineup {
  players: LineupPlayer[];
  total_salary: number;
  total_projected: number;
}

export function Optimize() {
  const [lineup, setLineup] = useState<Lineup | null>(null);
  const [loading, setLoading] = useState(false);
  const [salary, setSalary] = useState(50000);

  const generateLineup = async () => {
    setLoading(true);
    try {
      const response = await fetch(
        "http://localhost:8000/api/optimize/lineup",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            max_salary: salary,
            contest_type: "gpp",
          }),
        },
      );

      if (response.ok) {
        const data = await response.json();
        setLineup(data);
      } else {
        alert("Failed to generate lineup!");
      }
    } catch (error) {
      console.error("Optimization error:", error);
      alert("Failed to generate lineup!");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-6">
          <h1 className="text-3xl font-bold mb-4">Lineup Optimizer</h1>

          <Card className="mb-6">
            <CardHeader>
              <CardTitle>Configuration</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">
                  Max Salary: ${salary.toLocaleString()}
                </label>
                <Input
                  type="number"
                  value={salary}
                  onChange={(e) => setSalary(Number(e.target.value))}
                  min={0}
                  max={60000}
                />
              </div>
              <Button
                onClick={generateLineup}
                disabled={loading}
                className="w-full"
              >
                {loading ? "Generating..." : "Generate Optimal Lineup"}
              </Button>
            </CardContent>
          </Card>
        </div>

        {lineup && (
          <Card>
            <CardHeader>
              <CardTitle>Optimal Lineup</CardTitle>
              <CardDescription>
                Total Salary: ${lineup.total_salary.toLocaleString()} |
                Projected Points: {lineup.total_projected.toFixed(2)}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {lineup.players.map((player, index) => (
                  <div
                    key={index}
                    className="flex justify-between items-center p-3 border rounded"
                  >
                    <div>
                      <span className="font-medium">{player.name}</span>
                      <span className="text-sm text-muted-foreground ml-2">
                        ({player.position})
                      </span>
                    </div>
                    <div className="text-right">
                      <div className="font-medium">
                        ${player.salary.toLocaleString()}
                      </div>
                      <div className="text-sm text-muted-foreground">
                        {player.projected_points.toFixed(2)} pts
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
