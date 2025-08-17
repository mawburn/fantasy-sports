import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Link } from "react-router-dom";

export function Home() {
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold mb-4">NFL DFS Optimizer</h1>
        <p className="text-lg text-muted-foreground">
          Optimize your DraftKings lineups with ML-powered predictions
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Upload Salaries</CardTitle>
            <CardDescription>
              Upload DraftKings salary CSV files
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button asChild className="w-full">
              <Link to="/upload">Upload Salaries</Link>
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Generate Lineups</CardTitle>
            <CardDescription>
              Create optimized lineups for contests
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button asChild className="w-full">
              <Link to="/optimize">Optimize Lineups</Link>
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>View Predictions</CardTitle>
            <CardDescription>See ML predictions for players</CardDescription>
          </CardHeader>
          <CardContent>
            <Button asChild className="w-full">
              <Link to="/predictions">View Predictions</Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
