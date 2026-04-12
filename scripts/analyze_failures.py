"""
Analyse des modes d'échec de navigation.
Reproduce les seeds qui échouent et identifie les patterns.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
import re


def analyze_seed(seed, planner="bug2", early=False):
    """Analyse détaillée d'un seed problématique."""
    cmd = f"python scripts/benchmark_random.py --seed {seed} --episodes 1 --verbose --planner {parser}"
    if early:
        cmd += " --early-avoid on"
    
    result = subprocess.run(cmd.split(), capture_output=True, text=True, 
                           cwd="C:\\Users\\Bruno\\Documents\\GitHub\\DeskBot")
    
    output = result.stdout
    
    # Extraire les données
    fsm_changes = []
    front_distances = []
    positions = []
    
    for line in output.split('\n'):
        if 'fsm=' in line and 'front=' in line:
            # Parse: t=  2.0 fsm=go_heading   REAL=(+0.33,-0.00) h=-0 front=1.33
            match = re.search(r't=\s*([\d.]+).*fsm=(\w+).*REAL=\(([\d.+\-]+),([\d.+\-]+)\).*h=([\d.+\-]+).*front=([\d.]+|999.00)', line)
            if match:
                t, fsm, x, y, h, front = match.groups()
                fsm_changes.append((float(t), fsm, float(x), float(y), float(h), float(front) if front != '999.00' else None))
    
    return fsm_changes


def main():
    # Seeds qui échouent régulièrement
    test_seeds = [8147, 8161, 8164, 8127, 7685]
    
    print("=" * 80)
    print("ANALYSE DES MODES D'ECHEC")
    print("=" * 80)
    
    for seed in test_seeds:
        print(f"\n--- Seed {seed} ---")
        
        # Lancer le benchmark
        cmd = ["python", "scripts/benchmark_random.py", "--seed", str(seed), 
               "--episodes", "1", "--verbose", "--planner", "bug2"]
        
        result = subprocess.run(cmd, capture_output=True, text=True,
                               cwd="C:\\Users\\Bruno\\Documents\\GitHub\\DeskBot")
        
        lines = result.stdout.split('\n')
        
        # Afficher les 20 premières lignes de log
        for line in lines[10:30]:
            if 'fsm=' in line:
                print(line.strip())
        
        # Compter les transitions
        contour_count = sum(1 for l in lines if 'fsm=contour' in l)
        go_count = sum(1 for l in lines if 'fsm=go_heading' in l)
        reverse_count = sum(1 for l in lines if 'fsm=reverse' in l)
        
        print(f"  Transitions: {contour_count}x contour, {go_count}x go_heading, {reverse_count}x reverse")
        
        # Détecter les oscillations (alternance rapide)
        fsm_sequence = []
        for l in lines:
            if 'fsm=go_heading' in l:
                fsm_sequence.append('G')
            elif 'fsm=contour' in l:
                fsm_sequence.append('C')
        
        # Compter les transitions G->C->G
        oscillations = sum(1 for i in range(len(fsm_sequence)-2) 
                          if fsm_sequence[i:i+3] == ['G', 'C', 'G'])
        print(f"  Oscillations (G->C->G): {oscillations}")


if __name__ == "__main__":
    main()
