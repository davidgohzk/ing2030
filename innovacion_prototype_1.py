import math
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import json


class FlyerReceiverMatcher:
    """
    Advanced matching algorithm for pairing flyers with receivers based on multiple weighted factors.
    Uses a greedy optimization approach instead of linear programming for simplicity.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize the matcher with configurable weights.
        
        Args:
            weights: Dictionary of factor weights (should sum to 1.0)
        """
        self.default_weights = {
            "route": 0.35,      # Route matching importance
            "date": 0.25,       # Date flexibility importance
            "weight": 0.20,     # Weight/capacity utilization
            "price": 0.10,      # Price/budget alignment
            "reputation": 0.05, # Flyer reputation
            "urgency": 0.05     # Receiver urgency
        }
        
        self.weights = weights if weights else self.default_weights
        self._normalize_weights()
        
        self.match_results = None
        self.match_scores = {}
        
    def _normalize_weights(self):
        """Normalize weights to sum to 1.0"""
        total = sum(self.weights.values())
        if total > 0:
            for key in self.weights:
                self.weights[key] = self.weights[key] / total
    
    def route_score(self, flyer: Dict, receiver: Dict) -> float:
        """
        Score based on route matching (origin and destination).
        Returns 1.0 for perfect match, 0.3 for partial, 0 for no match.
        """
        origin_match = flyer["origin"].lower() == receiver["origin"].lower()
        dest_match = flyer["destination"].lower() == receiver["destination"].lower()
        
        if origin_match and dest_match:
            return 1.0
        elif origin_match or dest_match:
            return 0.3  # Partial match
        return 0.0
    
    def date_score(self, flyer: Dict, receiver: Dict, alpha: float = 0.3) -> float:
        """
        Score based on date proximity using exponential decay.
        
        Args:
            alpha: Decay rate (higher = stricter date matching)
        """
        try:
            f_date = datetime.strptime(flyer["date"], "%Y-%m-%d")
            r_date = datetime.strptime(receiver["date"], "%Y-%m-%d")
            days_diff = abs((f_date - r_date).days)
            
            # Exponential decay with cutoff
            if days_diff > 7:  # More than a week difference
                return 0.0
            return math.exp(-alpha * days_diff)
        except (ValueError, KeyError):
            return 0.0
    
    def weight_score(self, flyer: Dict, receiver: Dict, remaining_capacity: float = None) -> float:
        """
        Score based on weight utilization efficiency.
        """
        capacity = remaining_capacity if remaining_capacity is not None else flyer["capacity"]
        
        if capacity <= 0:
            return 0.0
            
        utilization = receiver["weight"] / capacity
        
        if utilization > 1.0:  # Over capacity
            return 0.0
        elif utilization < 0.3:  # Very low utilization
            return 0.3 + (utilization * 0.7 / 0.3)
        else:  # Good utilization (30-100%)
            return 0.7 + (utilization * 0.3)
    
    def price_score(self, flyer: Dict, receiver: Dict, beta: float = 0.05) -> float:
        """
        Score based on price-budget alignment using sigmoid function.
        """
        price_diff = flyer["price"] - receiver["budget"]
        
        if price_diff > 0:  # Over budget
            return max(0, 1 / (1 + math.exp(beta * price_diff * 2)))
        else:  # Within budget
            return min(1.0, 1 + 0.1 * (-price_diff / receiver["budget"]))
    
    def reputation_score(self, flyer: Dict) -> float:
        """
        Score based on flyer's reputation/rating.
        """
        rating = flyer.get("rating", 3.0)
        max_rating = 5.0
        normalized = rating / max_rating
        return normalized ** 1.5  # Emphasize high ratings
    
    def urgency_score(self, receiver: Dict) -> float:
        """
        Score based on receiver's urgency level.
        """
        urgency = receiver.get("urgency", "normal")
        urgency_map = {
            "low": 0.3,
            "normal": 0.6,
            "high": 0.9,
            "critical": 1.0
        }
        return urgency_map.get(urgency, 0.6)
    
    def calculate_match_score(self, flyer: Dict, receiver: Dict, remaining_capacity: float = None) -> Tuple[float, Dict[str, float]]:
        """
        Calculate overall match score and component scores.
        """
        scores = {
            "route": self.route_score(flyer, receiver),
            "date": self.date_score(flyer, receiver),
            "weight": self.weight_score(flyer, receiver, remaining_capacity),
            "price": self.price_score(flyer, receiver),
            "reputation": self.reputation_score(flyer),
            "urgency": self.urgency_score(receiver)
        }
        
        # Calculate weighted total
        total = sum(self.weights.get(k, 0) * scores[k] for k in scores)
        
        return total, scores
    
    def greedy_matching(self, 
                       flyers: List[Dict], 
                       receivers: List[Dict],
                       min_score_threshold: float = 0.3) -> Dict:
        """
        Perform greedy matching optimization.
        
        Args:
            flyers: List of flyer dictionaries
            receivers: List of receiver dictionaries
            min_score_threshold: Minimum score required for a match
        
        Returns:
            Dictionary containing matching results and metrics
        """
        # Initialize tracking variables
        flyer_capacities = {f["id"]: f["capacity"] for f in flyers}
        assigned_receivers = set()
        matches = []
        
        # Calculate all possible match scores
        all_matches = []
        for f in flyers:
            for r in receivers:
                score, components = self.calculate_match_score(f, r)
                if score >= min_score_threshold:
                    all_matches.append({
                        "flyer": f,
                        "receiver": r,
                        "score": score,
                        "components": components
                    })
        
        # Sort by score (descending)
        all_matches.sort(key=lambda x: x["score"], reverse=True)
        
        # Greedy assignment
        for match in all_matches:
            f_id = match["flyer"]["id"]
            r_id = match["receiver"]["id"]
            r_weight = match["receiver"]["weight"]
            
            # Check if receiver is already assigned or flyer has no capacity
            if r_id not in assigned_receivers and flyer_capacities[f_id] >= r_weight:
                # Make the assignment
                assigned_receivers.add(r_id)
                flyer_capacities[f_id] -= r_weight
                
                matches.append({
                    "flyer_id": f_id,
                    "receiver_id": r_id,
                    "total_score": match["score"],
                    "score_components": match["components"],
                    "weight_transferred": r_weight,
                    "price": match["flyer"]["price"],
                    "flyer_details": {
                        "origin": match["flyer"]["origin"],
                        "destination": match["flyer"]["destination"],
                        "date": match["flyer"]["date"],
                        "capacity": match["flyer"]["capacity"],
                        "rating": match["flyer"].get("rating", "N/A")
                    },
                    "receiver_details": {
                        "origin": match["receiver"]["origin"],
                        "destination": match["receiver"]["destination"],
                        "date": match["receiver"]["date"],
                        "weight": match["receiver"]["weight"],
                        "budget": match["receiver"]["budget"],
                        "urgency": match["receiver"].get("urgency", "normal")
                    }
                })
        
        # Calculate metrics
        total_capacity = sum(f["capacity"] for f in flyers)
        total_demand = sum(r["weight"] for r in receivers)
        total_weight_matched = sum(m["weight_transferred"] for m in matches)
        total_score = sum(m["total_score"] for m in matches)
        
        metrics = {
            "total_matches": len(matches),
            "total_score": total_score,
            "average_score": total_score / len(matches) if matches else 0,
            "capacity_utilization": (total_weight_matched / total_capacity * 100) if total_capacity > 0 else 0,
            "demand_satisfaction": (total_weight_matched / total_demand * 100) if total_demand > 0 else 0,
            "unmatched_receivers": len(receivers) - len(assigned_receivers)
        }
        
        self.match_results = {
            "success": True,
            "matches": matches,
            "metrics": metrics,
            "weights_used": self.weights
        }
        
        return self.match_results
    
    def print_results(self, detailed: bool = True):
        """
        Print matching results in a formatted way.
        """
        if not self.match_results:
            print("No results available. Run greedy_matching first.")
            return
        
        print("\n" + "="*80)
        print("✅ MATCHING RESULTS (Greedy Algorithm)")
        print("="*80)
        
        # Print matches
        print("\n📋 MATCHES:")
        print("-"*80)
        for match in self.match_results["matches"]:
            print(f"\n🎯 Flyer {match['flyer_id']} → Receiver {match['receiver_id']}")
            print(f"   Route: {match['flyer_details']['origin']} → {match['flyer_details']['destination']}")
            print(f"   Dates: Flyer [{match['flyer_details']['date']}] | Receiver [{match['receiver_details']['date']}]")
            print(f"   Weight: {match['weight_transferred']}kg / {match['flyer_details']['capacity']}kg capacity")
            print(f"   Price: ${match['price']} (Budget: ${match['receiver_details']['budget']})")
            print(f"   Match Score: {match['total_score']:.3f}")
            
            if detailed:
                print("   Score Breakdown:")
                for component, score in match['score_components'].items():
                    weight = self.weights.get(component, 0)
                    print(f"     - {component}: {score:.3f} (weight: {weight:.2f})")
        
        # Print metrics
        print("\n📊 METRICS:")
        print("-"*80)
        metrics = self.match_results["metrics"]
        print(f"Total Matches: {metrics['total_matches']}")
        print(f"Average Match Score: {metrics['average_score']:.3f}")
        print(f"Capacity Utilization: {metrics['capacity_utilization']:.1f}%")
        print(f"Demand Satisfaction: {metrics['demand_satisfaction']:.1f}%")
        print(f"Unmatched Receivers: {metrics['unmatched_receivers']}")
        
        print("\n" + "="*80)


# Example usage
if __name__ == "__main__":
    # Test data
    flyers = [
        {"id": 1, "origin": "Santiago", "destination": "Lima", "date": "2025-10-10", 
         "capacity": 10, "price": 50, "rating": 4.8},
        {"id": 2, "origin": "Santiago", "destination": "Buenos Aires", "date": "2025-10-09", 
         "capacity": 5, "price": 40, "rating": 4.2},
        {"id": 3, "origin": "Santiago", "destination": "Lima", "date": "2025-10-11", 
         "capacity": 8, "price": 55, "rating": 4.6},
        {"id": 4, "origin": "Santiago", "destination": "Lima", "date": "2025-10-10", 
         "capacity": 12, "price": 45, "rating": 4.9}
    ]
    
    receivers = [
        {"id": "A", "origin": "Santiago", "destination": "Lima", "date": "2025-10-10", 
         "weight": 8, "budget": 60, "urgency": "high"},
        {"id": "B", "origin": "Santiago", "destination": "Buenos Aires", "date": "2025-10-08", 
         "weight": 3, "budget": 45, "urgency": "normal"},
        {"id": "C", "origin": "Santiago", "destination": "Lima", "date": "2025-10-12", 
         "weight": 5, "budget": 40, "urgency": "low"},
        {"id": "D", "origin": "Santiago", "destination": "Lima", "date": "2025-10-10", 
         "weight": 7, "budget": 55, "urgency": "critical"},
        {"id": "E", "origin": "Santiago", "destination": "Lima", "date": "2025-10-11", 
         "weight": 4, "budget": 50, "urgency": "normal"}
    ]
    
    # Custom weights (optional)
    custom_weights = {
        "route": 0.40,      # Prioritize route matching
        "date": 0.20,       # Date flexibility
        "weight": 0.15,     # Weight efficiency
        "price": 0.10,      # Price alignment
        "reputation": 0.10, # Flyer reputation
        "urgency": 0.05     # Receiver urgency
    }
    
    # Initialize and run matcher
    matcher = FlyerReceiverMatcher(weights=custom_weights)
    
    print("🚀 Running Flyer-Receiver Matching Algorithm...")
    results = matcher.greedy_matching(flyers, receivers, min_score_threshold=0.4)
    
    # Print results
    matcher.print_results(detailed=True)
