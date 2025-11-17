// Disk Scheduling Algorithms Simulator
// Assignment III - Operating Systems 2025-2
// Implements FCFS, SCAN, and C-SCAN algorithms

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <map>
#include <tuple>

using namespace std;

class DiskScheduler {
private:
    int num_cylinders;
    int num_requests;
    vector<int> requests;

public:
    DiskScheduler(int cylinders = 5000, int num_req = 1000) 
        : num_cylinders(cylinders), num_requests(num_req) {}

    void generateRequests() {
        srand(time(nullptr));
        
        requests.clear();
        for (int i = 0; i < num_requests; i++) {
            requests.push_back(rand() % num_cylinders);
        }
    }

    vector<int> getRequests() const {
        return requests;
    }

    // FCFS algorithm
    int fcfs(int initial_position, vector<int>& sequence) {
        sequence.clear();
        sequence.push_back(initial_position);
        
        int total_movement = 0;
        int current_position = initial_position;
        
        for (int request : requests) {
            int movement = abs(request - current_position);
            total_movement += movement;
            current_position = request;
            sequence.push_back(current_position);
        }
        
        return total_movement;
    }

    // SCAN algorithm
    int scan(int initial_position, vector<int>& sequence, string direction = "up") {
        sequence.clear();
        sequence.push_back(initial_position);
        
        int total_movement = 0;
        int current_position = initial_position;
        
        // Separate requests into lower and upper
        vector<int> lower, upper;
        for (int request : requests) {
            if (request < current_position) {
                lower.push_back(request);
            } else {
                upper.push_back(request);
            }
        }
        
        // Sort the vectors
        sort(lower.begin(), lower.end(), greater<int>()); // Descending
        sort(upper.begin(), upper.end());                  // Ascending
        
        if (direction == "up") {
            // Service upper requests first, moving up
            for (int request : upper) {
                int movement = abs(request - current_position);
                total_movement += movement;
                current_position = request;
                sequence.push_back(current_position);
            }
            
            // Move to the end if there were upper requests
            if (!upper.empty() && upper.back() != num_cylinders - 1) {
                int movement = (num_cylinders - 1) - current_position;
                total_movement += movement;
                current_position = num_cylinders - 1;
                sequence.push_back(current_position);
            }
            
            // Service lower requests, moving down
            for (int request : lower) {
                int movement = abs(request - current_position);
                total_movement += movement;
                current_position = request;
                sequence.push_back(current_position);
            }
        } else { // direction == "down"
            // Service lower requests first, moving down
            for (int request : lower) {
                int movement = abs(request - current_position);
                total_movement += movement;
                current_position = request;
                sequence.push_back(current_position);
            }
            
            // Move to the beginning if there were lower requests
            if (!lower.empty() && lower.back() != 0) {
                int movement = current_position;
                total_movement += movement;
                current_position = 0;
                sequence.push_back(current_position);
            }
            
            // Service upper requests, moving up
            for (int request : upper) {
                int movement = abs(request - current_position);
                total_movement += movement;
                current_position = request;
                sequence.push_back(current_position);
            }
        }
        
        return total_movement;
    }

    // C-SCAN algorithm
    int cscan(int initial_position, vector<int>& sequence) {
        sequence.clear();
        sequence.push_back(initial_position);
        
        int total_movement = 0;
        int current_position = initial_position;
        
        // Separate requests into lower and upper
        vector<int> lower, upper;
        for (int request : requests) {
            if (request < current_position) {
                lower.push_back(request);
            } else {
                upper.push_back(request);
            }
        }
        
        // Sort the vectors
        sort(lower.begin(), lower.end());  // Ascending
        sort(upper.begin(), upper.end());  // Ascending
        
        // Service upper requests first, moving up
        for (int request : upper) {
            int movement = abs(request - current_position);
            total_movement += movement;
            current_position = request;
            sequence.push_back(current_position);
        }
        
        // Move to the end if we serviced any upper requests
        if (!upper.empty() && current_position != num_cylinders - 1) {
            int movement = (num_cylinders - 1) - current_position;
            total_movement += movement;
            current_position = num_cylinders - 1;
            sequence.push_back(current_position);
        }
        
        // Jump to the beginning (this counts as movement)
        if (!lower.empty()) {
            int movement = current_position; // Distance to cylinder 0
            total_movement += movement;
            current_position = 0;
            sequence.push_back(current_position);
            
            // Service lower requests, moving up
            for (int request : lower) {
                int movement = abs(request - current_position);
                total_movement += movement;
                current_position = request;
                sequence.push_back(current_position);
            }
        }
        
        return total_movement;
    }

    void saveSequences(int initial_position, const vector<int>& fcfs_seq,
                      const vector<int>& scan_seq, const vector<int>& cscan_seq) {
        ofstream file("sequences.txt");
        
        if (!file.is_open()) {
            cerr << "Error: No se pudo crear sequences.txt" << endl;
            return;
        }
        
        // Write initial position
        file << initial_position << endl;
        
        // Write requests
        for (size_t i = 0; i < requests.size(); i++) {
            if (i > 0) file << ",";
            file << requests[i];
        }
        file << endl;
        
        // Write FCFS sequence
        for (size_t i = 0; i < fcfs_seq.size(); i++) {
            if (i > 0) file << ",";
            file << fcfs_seq[i];
        }
        file << endl;
        
        // Write SCAN sequence
        for (size_t i = 0; i < scan_seq.size(); i++) {
            if (i > 0) file << ",";
            file << scan_seq[i];
        }
        file << endl;
        
        // Write C-SCAN sequence
        for (size_t i = 0; i < cscan_seq.size(); i++) {
            if (i > 0) file << ",";
            file << cscan_seq[i];
        }
        file << endl;
        
        file.close();
    }

    void saveResults(int initial_position, int fcfs_movement,
                    int scan_movement, int cscan_movement) {
        ofstream file("disk_scheduling_results.txt");
        
        if (!file.is_open()) {
            cerr << "Error: No se pudo crear disk_scheduling_results.txt" << endl;
            return;
        }
        
        file << string(80, '=') << endl;
        file << "ALGORITMOS DE PLANIFICACION DE DISCO - RESULTADOS DETALLADOS" << endl;
        file << string(80, '=') << endl;
        file << "Posicion inicial del cabezal: " << initial_position << endl;
        file << "Numero de solicitudes: " << requests.size() << endl;
        file << endl;
        
        file << "Todas las solicitudes:" << endl;
        for (size_t i = 0; i < requests.size(); i += 10) {
            file << "[";
            for (size_t j = i; j < min(i + 10, requests.size()); j++) {
                if (j > i) file << ", ";
                file << requests[j];
            }
            file << "]" << endl;
        }
        file << endl;
        
        file << "Movimiento total FCFS: " << fcfs_movement << " cilindros" << endl;
        file << "Movimiento total SCAN: " << scan_movement << " cilindros" << endl;
        file << "Movimiento total C-SCAN: " << cscan_movement << " cilindros" << endl;
        
        file.close();
    }
};

string formatNumber(int num) {
    string s = to_string(num);
    int n = s.length() - 3;
    while (n > 0) {
        s.insert(n, ",");
        n -= 3;
    }
    return s;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Uso: " << argv[0] << " <posicion_inicial_cabezal>" << endl;
        cout << "Ejemplo: " << argv[0] << " 2500" << endl;
        return 1;
    }
    
    int initial_position = atoi(argv[1]);
    
    // Validate initial position
    const int NUM_CYLINDERS = 5000;
    if (initial_position < 0 || initial_position >= NUM_CYLINDERS) {
        cerr << "Error: La posicion inicial debe estar entre 0 y " 
             << NUM_CYLINDERS - 1 << endl;
        return 1;
    }
    
    // Create scheduler and generate requests
    DiskScheduler scheduler(NUM_CYLINDERS, 1000);
    scheduler.generateRequests();
    vector<int> requests = scheduler.getRequests();
    
    cout << string(80, '=') << endl;
    cout << "SIMULACION DE ALGORITMOS DE PLANIFICACION DE DISCO" << endl;
    cout << string(80, '=') << endl;
    cout << "Numero de cilindros: " << NUM_CYLINDERS 
         << " (0 a " << NUM_CYLINDERS - 1 << ")" << endl;
    cout << "Numero de solicitudes: " << requests.size() << endl;
    cout << "Posicion inicial del cabezal: " << initial_position << endl;
    cout << endl;
    
    // Show first 20 requests
    cout << "Primeras 20 solicitudes: [";
    for (int i = 0; i < min(20, (int)requests.size()); i++) {
        if (i > 0) cout << ", ";
        cout << requests[i];
    }
    cout << "]" << endl << endl;
    
    // Run FCFS
    cout << string(80, '-') << endl;
    cout << "1. FCFS (Primero en Llegar, Primero en Ser Atendido)" << endl;
    cout << string(80, '-') << endl;
    vector<int> fcfs_sequence;
    int fcfs_movement = scheduler.fcfs(initial_position, fcfs_sequence);
    cout << "Movimiento total del cabezal: " << formatNumber(fcfs_movement) 
         << " cilindros" << endl;
    cout << "Distancia promedio de busqueda: " << fixed << setprecision(2)
         << (double)fcfs_movement / requests.size() 
         << " cilindros por solicitud" << endl << endl;
    
    // Run SCAN
    cout << string(80, '-') << endl;
    cout << "2. SCAN (Algoritmo del Elevador - moviendose hacia arriba)" << endl;
    cout << string(80, '-') << endl;
    vector<int> scan_sequence;
    int scan_movement = scheduler.scan(initial_position, scan_sequence, "up");
    cout << "Movimiento total del cabezal: " << formatNumber(scan_movement) 
         << " cilindros" << endl;
    cout << "Distancia promedio de busqueda: " << fixed << setprecision(2)
         << (double)scan_movement / requests.size() 
         << " cilindros por solicitud" << endl;
    double improvement_scan = ((double)(fcfs_movement - scan_movement) / fcfs_movement) * 100;
    cout << "Mejora sobre FCFS: " << fixed << setprecision(2)
         << improvement_scan << "%" << endl << endl;
    
    // Run C-SCAN
    cout << string(80, '-') << endl;
    cout << "3. C-SCAN (SCAN Circular)" << endl;
    cout << string(80, '-') << endl;
    vector<int> cscan_sequence;
    int cscan_movement = scheduler.cscan(initial_position, cscan_sequence);
    cout << "Movimiento total del cabezal: " << formatNumber(cscan_movement) 
         << " cilindros" << endl;
    cout << "Distancia promedio de busqueda: " << fixed << setprecision(2)
         << (double)cscan_movement / requests.size() 
         << " cilindros por solicitud" << endl;
    double improvement_cscan = ((double)(fcfs_movement - cscan_movement) / fcfs_movement) * 100;
    cout << "Mejora sobre FCFS: " << fixed << setprecision(2)
         << improvement_cscan << "%" << endl << endl;
    
    // Summary comparison
    cout << string(80, '=') << endl;
    cout << "RESUMEN COMPARATIVO" << endl;
    cout << string(80, '=') << endl;
    cout << left << setw(15) << "Algoritmo" 
         << setw(25) << "Movimiento Total" 
         << setw(20) << "Promedio/Solicitud" 
         << "Ranking" << endl;
    cout << string(80, '-') << endl;
    
    // Determine rankings
    vector<pair<string, int>> results = {
        {"FCFS", fcfs_movement},
        {"SCAN", scan_movement},
        {"C-SCAN", cscan_movement}
    };
    sort(results.begin(), results.end(), 
         [](const pair<string, int>& a, const pair<string, int>& b) {
             return a.second < b.second;
         });
    
    map<string, int> rankings;
    for (size_t i = 0; i < results.size(); i++) {
        rankings[results[i].first] = i + 1;
    }
    
    // Print results
    vector<tuple<string, int, double>> all_results = {
        make_tuple("FCFS", fcfs_movement, (double)fcfs_movement / requests.size()),
        make_tuple("SCAN", scan_movement, (double)scan_movement / requests.size()),
        make_tuple("C-SCAN", cscan_movement, (double)cscan_movement / requests.size())
    };
    
    for (const auto& result : all_results) {
        string name = get<0>(result);
        int total = get<1>(result);
        double avg = get<2>(result);
        int rank = rankings[name];
        
        cout << left << setw(15) << name
             << right << setw(10) << formatNumber(total) << " cilindros     "
             << setw(8) << fixed << setprecision(2) << avg << " cilindros   "
             << "#" << rank << endl;
    }
    
    cout << string(80, '=') << endl;
    
    // Save results
    scheduler.saveResults(initial_position, fcfs_movement, 
                         scan_movement, cscan_movement);
    scheduler.saveSequences(initial_position, fcfs_sequence, 
                           scan_sequence, cscan_sequence);
    
    cout << "\nResultados detallados guardados en: disk_scheduling_results.txt" << endl;
    cout << "Datos de secuencia guardados en: sequences.txt" << endl;
    
    return 0;
}
