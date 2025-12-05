/**
 * MOCK BACKEND: get_controller(scenario_text)
 * Returns the JSON structure of the FSA.
 */
function get_controller(scenarioId) {
    // Base template for the FSA graph
    const fsa = {
        "scenario_id": scenarioId,
        "graph": {
            "nodes": [
                { "id": 0, "description": "Initial" },
                { "id": 1, "description": "Action: go_straight" },
                { "id": 2, "description": "Action: stop" },
                { "id": 3, "description": "Action: wait" },
                { "id": 4, "description": "Action: turn_left" },
                { "id": 5, "description": "Action: turn_right" }
            ],
            "edges": []
        }
    };

    const edges = fsa.graph.edges;

    // Logic to build edges based on the scenario intent
    switch(parseInt(scenarioId)) {
        case 1: // Green light + Ped Front -> Stop, then Wait/Go
            edges.push(
                { from: 0, to: 2, condition: "pedestrian_in_front", action: "stop" },
                { from: 2, to: 3, condition: "pedestrian_in_front", action: "wait" },
                { from: 3, to: 1, condition: "!pedestrian_in_front", action: "go_straight" }
            );
            break;
        case 2: // Turn Right blocked by Ped
            edges.push(
                { from: 0, to: 2, condition: "pedestrian_at_right", action: "stop" },
                { from: 2, to: 3, condition: "pedestrian_at_right", action: "wait" },
                { from: 3, to: 5, condition: "!pedestrian_at_right", action: "turn_right" }
            );
            break;
        case 3: // Turn Left blocked by Ped
            edges.push(
                { from: 0, to: 2, condition: "pedestrian_at_left", action: "stop" },
                { from: 2, to: 3, condition: "pedestrian_at_left", action: "wait" },
                { from: 3, to: 4, condition: "!pedestrian_at_left", action: "turn_left" }
            );
            break;
        case 4: // Left Turn blocked by Opposing Car
            edges.push(
                { from: 0, to: 3, condition: "opposite_car", action: "wait" },
                { from: 3, to: 4, condition: "!opposite_car", action: "turn_left" }
            );
            break;
        case 5: // Unprotected Left Turn
            edges.push(
                { from: 0, to: 3, condition: "car_from_left", action: "wait" },
                { from: 3, to: 4, condition: "!car_from_left", action: "turn_left" }
            );
            break;
        case 7: // Protected Left
            edges.push(
                { from: 0, to: 4, condition: "green_left_turn_light", action: "turn_left" }
            );
            break;
        case 8: // Red light right turn on clear
            edges.push(
                { from: 0, to: 2, condition: "true", action: "stop" }, // Full stop first
                { from: 2, to: 5, condition: "!car_from_left && !pedestrian_at_right", action: "turn_right" }
            );
            break;
        case 10: // Stop sign cross traffic
            edges.push(
                { from: 0, to: 2, condition: "stop_sign", action: "stop" },
                { from: 2, to: 3, condition: "car_from_left || car_from_right", action: "wait" },
                { from: 3, to: 1, condition: "!car_from_left && !car_from_right", action: "go_straight" } 
            );
            break;
        case 13: // Red light straight
            edges.push(
                { from: 0, to: 2, condition: "!green_traffic_light", action: "stop" },
                { from: 2, to: 3, condition: "!green_traffic_light", action: "wait" },
                { from: 3, to: 1, condition: "green_traffic_light", action: "go_straight" }
            );
            break;
        case 11: // Stop sign no traffic
            edges.push(
                 { from: 0, to: 2, condition: "stop_sign", action: "stop" },
                 { from: 2, to: 1, condition: "true", action: "go_straight" }
            );
            break;
        case 14: // Green light but currently stopped
             edges.push(
                 { from: 0, to: 1, condition: "green_traffic_light", action: "go_straight" }
             )
             break;
        default:
            // Generic safety fallback for other scenarios
            edges.push(
                { from: 0, to: 3, condition: "obstacle_detected", action: "wait" },
                { from: 0, to: 1, condition: "clear_path", action: "go_straight" }
            );
            break;
    }

    return JSON.stringify(fsa, null, 2);
}
