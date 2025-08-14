import Papa from "papaparse";
import { promises as fs } from "fs";
import path from "path";

// Default CSV file - change this to your CSV filename
const DEFAULT_CSV = "DKSalaries.csv";

// Script 1: Initial CSV Data Exploration
async function exploreCSVData(filename = DEFAULT_CSV) {
  // Read the CSV file using Node.js file system
  const csvData = await fs.readFile(path.resolve(filename), "utf8");

  const parsed = Papa.parse(csvData, {
    header: true,
    dynamicTyping: true, // Automatically convert numbers
    skipEmptyLines: true,
  });

  console.log("Total players:", parsed.data.length);
  console.log("Sample data:", parsed.data[0]);

  // Analyze roster positions available
  const positions = parsed.data
    .map((p) => p["Roster Position"])
    .filter((p) => p);
  const uniquePositions = [...new Set(positions)];
  console.log("Unique roster positions:", uniquePositions);

  // Count players by position
  const positionCounts = {};
  positions.forEach((pos) => {
    positionCounts[pos] = (positionCounts[pos] || 0) + 1;
  });
  console.log("Position counts:", positionCounts);
}

// Script 2: Player Analysis by Position and Team
async function analyzePlayersByTeam(filename = DEFAULT_CSV) {
  const csvData = await fs.readFile(path.resolve(filename), "utf8");

  const parsed = Papa.parse(csvData, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
  });

  const captains = parsed.data.filter((p) => p["Roster Position"] === "CPT");
  const flex = parsed.data.filter((p) => p["Roster Position"] === "FLEX");

  console.log("Top 10 Captain options by salary:");
  captains
    .sort((a, b) => b.Salary - a.Salary)
    .slice(0, 10)
    .forEach((p) => {
      console.log(
        `${p.Name} (${p.TeamAbbrev}) - $${p.Salary} - ${p.AvgPointsPerGame} FPPG`
      );
    });

  console.log("Top 10 FLEX options by salary:");
  flex
    .sort((a, b) => b.Salary - a.Salary)
    .slice(0, 10)
    .forEach((p) => {
      console.log(
        `${p.Name} (${p.TeamAbbrev}) - $${p.Salary} - ${p.AvgPointsPerGame} FPPG`
      );
    });
}

// Script 3: Smart Lineup Builder with Value Analysis
async function buildOptimalLineups(filename = DEFAULT_CSV, salaryCap = 50000) {
  const csvData = await fs.readFile(path.resolve(filename), "utf8");

  const parsed = Papa.parse(csvData, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
  });

  const flex = parsed.data.filter((p) => p["Roster Position"] === "FLEX");

  // Find the BEST cheap options, not just minimum salary
  const budgetOptions = flex
    .filter((p) => p.Salary <= 3000 && p.AvgPointsPerGame > 0)
    .sort((a, b) => b.AvgPointsPerGame - a.AvgPointsPerGame);

  console.log("BEST budget options for lineup construction:");
  budgetOptions.slice(0, 8).forEach((p) => {
    console.log(
      `${p.Name} (${p.TeamAbbrev}) ${p.Position} - ${p.Salary} - ${p.AvgPointsPerGame} FPPG`
    );
  });

  // Build sample lineups with SMART value picks
  console.log("\n=== STRATEGY 1: Lamar Jackson Captain (IMPROVED) ===");
  const lineup1 = [
    {
      name: "Lamar Jackson (CPT)",
      salary: 18300,
      fppg: 24.26,
      team: "BAL",
      pos: "QB",
    },
    {
      name: "Tony Pollard (FLEX)",
      salary: 10400,
      fppg: 19.17,
      team: "TEN",
      pos: "RB",
    },
    {
      name: "Calvin Ridley (FLEX)",
      salary: 8400,
      fppg: 12.8,
      team: "TEN",
      pos: "WR",
    },
    {
      name: "Tyler Boyd (FLEX)",
      salary: 7000,
      fppg: 9.38,
      team: "TEN",
      pos: "WR",
    },
    {
      name: "Ravens DST (FLEX)",
      salary: 4000,
      fppg: 4.46,
      team: "BAL",
      pos: "DST",
    },
    // Use BEST remaining value instead of minimum salary
    budgetOptions.length > 0
      ? {
          name: `${budgetOptions[0].Name} (FLEX)`,
          salary: budgetOptions[0].Salary,
          fppg: budgetOptions[0].AvgPointsPerGame,
          team: budgetOptions[0].TeamAbbrev,
          pos: budgetOptions[0].Position,
        }
      : {
          name: "Filler (FLEX)",
          salary: 1000,
          fppg: 0,
          team: "MIN",
          pos: "FILLER",
        },
  ];

  const total = lineup1.reduce((sum, p) => sum + p.salary, 0);
  const projectedPoints = lineup1.reduce((sum, p) => {
    const multiplier = p.name.includes("(CPT)") ? 1.5 : 1;
    return sum + p.fppg * multiplier;
  }, 0);

  console.log("Total cost:", total);
  console.log("Under budget by:", salaryCap - total);
  console.log("Projected points:", projectedPoints.toFixed(2));

  lineup1.forEach((p) =>
    console.log(`${p.name}: ${p.salary} (${p.fppg} FPPG)`)
  );

  // Show why value picks matter
  console.log("\n=== VALUE COMPARISON ===");
  if (budgetOptions.length > 0) {
    const bestBudget = budgetOptions[0];
    const minSalary = { name: "Min Salary Player", salary: 1000, fppg: 0 };
    console.log(
      `Best Budget: ${bestBudget.Name} - ${bestBudget.Salary} - ${bestBudget.AvgPointsPerGame} FPPG`
    );
    console.log(
      `Min Salary: ${minSalary.name} - ${minSalary.salary} - ${minSalary.fppg} FPPG`
    );
    console.log(`Extra cost: ${bestBudget.Salary - minSalary.salary}`);
    console.log(
      `Extra points: ${bestBudget.AvgPointsPerGame - minSalary.fppg}`
    );
    console.log(
      `Points per extra dollar: ${(
        ((bestBudget.AvgPointsPerGame - minSalary.fppg) /
          (bestBudget.Salary - minSalary.salary)) *
        1000
      ).toFixed(2)} pts/$1k`
    );
  }

  return lineup1; // Return the lineup for further use
}

// Script 4: Advanced Value Analysis (FIXED)
async function findValuePlayers(filename = DEFAULT_CSV) {
  const csvData = await fs.readFile(path.resolve(filename), "utf8");

  const parsed = Papa.parse(csvData, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
  });

  const flex = parsed.data.filter((p) => p["Roster Position"] === "FLEX");

  // Find cheap options for lineup construction - SORTED BY FPPG!
  const cheapFlex = flex
    .filter((p) => p.Salary <= 4000)
    .sort((a, b) => b.AvgPointsPerGame - a.AvgPointsPerGame);

  console.log("Cheap FLEX options ($4000 and under) - SORTED BY FPPG:");
  cheapFlex.slice(0, 15).forEach((p) => {
    console.log(
      `${p.Name} (${p.TeamAbbrev}) ${p.Position} - ${p.Salary} - ${p.AvgPointsPerGame} FPPG`
    );
  });

  // Calculate points per dollar value (but exclude 0 FPPG players)
  const valueAnalysis = flex
    .filter((p) => p.AvgPointsPerGame > 0) // Only players with actual production
    .map((p) => ({
      ...p,
      pointsPerDollar: (p.AvgPointsPerGame / p.Salary) * 1000, // Points per $1000 spent
    }))
    .sort((a, b) => b.pointsPerDollar - a.pointsPerDollar);

  console.log("\nBest value plays (points per $1000) - EXCLUDING 0 FPPG:");
  valueAnalysis.slice(0, 15).forEach((p) => {
    console.log(
      `${p.Name} (${p.TeamAbbrev}) ${p.Position} - ${p.Salary} - ${
        p.AvgPointsPerGame
      } FPPG - ${p.pointsPerDollar.toFixed(2)} pts/$1k`
    );
  });

  // Specific analysis for sub-$3000 options with actual upside
  const budgetOptions = flex
    .filter((p) => p.Salary <= 3000 && p.AvgPointsPerGame > 0)
    .sort((a, b) => b.AvgPointsPerGame - a.AvgPointsPerGame);

  console.log("\nBest budget options under $3000 with actual upside:");
  budgetOptions.forEach((p) => {
    console.log(
      `${p.Name} (${p.TeamAbbrev}) ${p.Position} - ${p.Salary} - ${p.AvgPointsPerGame} FPPG`
    );
  });
}

// Script 5: Team-based Analysis
async function analyzeByTeam(filename = DEFAULT_CSV) {
  const csvData = await fs.readFile(path.resolve(filename), "utf8");

  const parsed = Papa.parse(csvData, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
  });

  // Get unique teams
  const teams = [...new Set(parsed.data.map((p) => p.TeamAbbrev))];

  teams.forEach((team) => {
    console.log(`\n=== ${team} PLAYERS ===`);
    const teamPlayers = parsed.data
      .filter((p) => p.TeamAbbrev === team && p.AvgPointsPerGame > 5)
      .sort((a, b) => b.AvgPointsPerGame - a.AvgPointsPerGame);

    teamPlayers.forEach((p) => {
      const cptPrice =
        parsed.data.find(
          (c) => c.Name === p.Name && c["Roster Position"] === "CPT"
        )?.Salary || "N/A";
      const flexPrice =
        parsed.data.find(
          (f) => f.Name === p.Name && f["Roster Position"] === "FLEX"
        )?.Salary || "N/A";

      console.log(
        `${p.Position} ${p.Name} - CPT: $${cptPrice} | FLEX: $${flexPrice} - ${p.AvgPointsPerGame} FPPG`
      );
    });
  });
}

// Enhanced lineup validation with value analysis
function validateLineup(lineup, salaryCap = 50000) {
  const totalSalary = lineup.reduce((sum, player) => sum + player.salary, 0);
  const projectedPoints = lineup.reduce((sum, player) => {
    // Captain gets 1.5x multiplier
    const multiplier = player.name.includes("(CPT)") ? 1.5 : 1;
    return sum + player.fppg * multiplier;
  }, 0);

  // Calculate if we have room for upgrades
  const remainingBudget = salaryCap - totalSalary;
  const hasValueOpportunity = remainingBudget >= 500; // If we have $500+ left, look for upgrades

  return {
    totalSalary,
    remainingBudget,
    projectedPoints: projectedPoints.toFixed(2),
    isValid: totalSalary <= salaryCap && lineup.length === 6,
    hasValueOpportunity,
    suggestions: hasValueOpportunity
      ? "Consider upgrading lowest FPPG players with remaining budget"
      : "Lineup optimized for salary cap",
  };
}

// New function: Find optimal value upgrades for remaining budget
function findUpgradeOptions(currentPlayer, budget, allPlayers) {
  const upgrades = allPlayers
    .filter(
      (p) =>
        p.Salary > currentPlayer.salary &&
        p.Salary <= currentPlayer.salary + budget &&
        p.Position === currentPlayer.pos && // Same position
        p.AvgPointsPerGame > currentPlayer.fppg // Better production
    )
    .map((p) => ({
      ...p,
      extraCost: p.Salary - currentPlayer.salary,
      extraPoints: p.AvgPointsPerGame - currentPlayer.fppg,
      efficiency:
        ((p.AvgPointsPerGame - currentPlayer.fppg) /
          (p.Salary - currentPlayer.salary)) *
        1000,
    }))
    .sort((a, b) => b.efficiency - a.efficiency);

  return upgrades.slice(0, 5); // Top 5 upgrade options
}

// Usage examples - how to run the generic functions
async function runAnalysis(csvFile = DEFAULT_CSV) {
  // Example 1: Analyze the game
  console.log(`=== ANALYZING ${csvFile} ===`);
  await buildOptimalLineups(csvFile, 50000);

  // Example 2: Find value players
  console.log("\n=== VALUE ANALYSIS ===");
  await findValuePlayers(csvFile);

  // Example 3: Compare different salary caps
  console.log("\n=== DIFFERENT SALARY CAP ANALYSIS ===");
  await buildOptimalLineups(csvFile, 60000); // Higher cap
}

// Generic function that works with any showdown CSV
async function analyzeAnyShowdownCSV(
  filename = DEFAULT_CSV,
  salaryCap = 50000
) {
  console.log(`=== ANALYZING ${filename} ===`);

  // Step 1: Explore the data
  await exploreCSVData(filename);

  // Step 2: Build optimal lineups
  const strategies = await buildOptimalLineups(filename, salaryCap);

  // Step 3: Find value plays
  await findValuePlayers(filename);

  // Step 4: Team analysis
  await analyzeByTeam(filename);

  return strategies;
}

// Export functions for reuse
export {
  exploreCSVData,
  analyzePlayersByTeam,
  buildOptimalLineups,
  findValuePlayers,
  analyzeByTeam,
  validateLineup,
  findUpgradeOptions,
  runAnalysis,
  analyzeAnyShowdownCSV,
};
