import cv2
import numpy as np
import random
import time
from typing import List, Dict, Any
from src.types import FrameData, TrackedPerson

class MockPersonSim:
    def __init__(self, person_id: int, x: float, y: float, has_helmet: bool = True, has_glasses: bool = True):
        self.person_id = person_id
        self.x = x
        self.y = y
        self.vx = random.choice([-2.0, -1.0, 1.0, 2.0])
        self.vy = random.choice([-1.0, -0.5, 0.5, 1.0])
        self.w = 60
        self.h = 140
        self.has_helmet = has_helmet
        self.has_glasses = has_glasses
        self.is_fallen = False
        self.has_exposed_tattoo = random.choice([True, False, False, False])
        self.time_fallen = 0.0

    def update(self, width: int, height: int):
        if self.is_fallen:
            # When fallen, person stays still and changes aspect ratio
            self.w = 140
            self.h = 60
            return True # Still active in frame

        # Move
        self.x += self.vx
        self.y += self.vy
        
        # Check boundary collision or exit
        # We allow them to leave the screen with a 2% chance if they are close to boundaries
        margin = 10
        if (self.x < margin or self.x > width - self.w - margin or 
            self.y < margin or self.y > height - self.h - margin):
            if random.random() < 0.01:
                return False # Walks off-screen (deactivate)
            
            # Normal bounce
            if self.x < 0 or self.x > width - self.w:
                self.vx *= -1
            if self.y < 0 or self.y > height - self.h:
                self.vy *= -1
                
        self.x = max(0, min(self.x, width - self.w))
        self.y = max(0, min(self.y, height - self.h))
        return True

class MockPipelineGenerator:
    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.active_sims: List[MockPersonSim] = []
        self.past_visitors: List[MockPersonSim] = []
        self.next_id = 1
        self.frame_index = 0
        self.smoke_active = False
        self.smoke_particles = []
        
        # Spawn initial workers
        for _ in range(3):
            self._spawn_worker()

    def _spawn_worker(self):
        # 30% chance to bring back a past visitor (demonstrating ReID)
        if len(self.past_visitors) > 0 and random.random() < 0.3:
            visitor = self.past_visitors.pop(random.randint(0, len(self.past_visitors) - 1))
            visitor.x = random.choice([10.0, float(self.width - visitor.w - 10)])
            visitor.y = float(random.randint(50, self.height - visitor.h - 50))
            visitor.vx = 2.0 if visitor.x < self.width / 2 else -2.0
            visitor.vy = random.choice([-1.0, 1.0])
            visitor.is_fallen = False
            visitor.w = 60
            visitor.h = 140
            self.active_sims.append(visitor)
            print(f"[MockEngine] Worker {visitor.person_id} re-entered the room (Re-ID Match!)")
        else:
            # Create a brand new worker
            pid = self.next_id
            self.next_id += 1
            x = float(random.randint(50, self.width - 100))
            y = float(random.randint(50, self.height - 200))
            
            # Make some workers non-compliant
            has_helmet = random.random() > 0.25
            has_glasses = random.random() > 0.2
            
            worker = MockPersonSim(pid, x, y, has_helmet, has_glasses)
            self.active_sims.append(worker)
            print(f"[MockEngine] New Worker {pid} entered the room.")

    def update_simulation(self):
        # Update workers
        still_active = []
        for worker in self.active_sims:
            keep = worker.update(self.width, self.height)
            if keep:
                still_active.append(worker)
            else:
                # Cache their visual profile for Re-ID when they return
                self.past_visitors.append(worker)
                print(f"[MockEngine] Worker {worker.person_id} left the room.")
                
        self.active_sims = still_active

        # Randomly spawn workers to maintain population between 1 and 4
        if len(self.active_sims) < 2 or (len(self.active_sims) < 4 and random.random() < 0.005):
            self._spawn_worker()

        # Randomly trigger a fall
        if random.random() < 0.002 and len(self.active_sims) > 0:
            unfallen = [w for w in self.active_sims if not w.is_fallen]
            if unfallen:
                target = random.choice(unfallen)
                target.is_fallen = True
                target.time_fallen = time.time()
                print(f"[MockEngine] Worker {target.person_id} has fallen down!")

        # Recover workers from falls after 8 seconds
        for worker in self.active_sims:
            if worker.is_fallen and (time.time() - worker.time_fallen > 8.0):
                worker.is_fallen = False
                worker.w = 60
                worker.h = 140
                print(f"[MockEngine] Worker {worker.person_id} got back up.")

        # Toggle smoke environmental hazard every 200 frames
        if self.frame_index % 250 == 0 and self.frame_index > 0:
            self.smoke_active = not self.smoke_active
            print(f"[MockEngine] Smoke simulation: {'ENABLED' if self.smoke_active else 'DISABLED'}")

    def draw_scene(self) -> np.ndarray:
        # Create a warehouse background frame
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Draw background elements (grid floor, conveyor belt lines)
        cv2.rectangle(frame, (0, 0), (self.width, self.height), (30, 25, 25), -1)
        for i in range(0, self.height, 40):
            cv2.line(frame, (0, i), (self.width, i), (40, 35, 35), 1)
        for j in range(0, self.width, 40):
            cv2.line(frame, (j, 0), (j, self.height), (40, 35, 35), 1)

        # Draw a simulated industrial machine area
        cv2.rectangle(frame, (50, 50), (180, 200), (60, 60, 50), -1)
        cv2.putText(frame, "Curing Oven", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        
        cv2.rectangle(frame, (450, 250), (600, 420), (50, 60, 60), -1)
        cv2.putText(frame, "CNC Mill", (470, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

        # Draw simulated safety hazard zone
        cv2.rectangle(frame, (250, 150), (390, 280), (0, 0, 100), 2)
        cv2.putText(frame, "RESTRICTED", (270, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw workers
        for worker in self.active_sims:
            x, y, w, h = int(worker.x), int(worker.y), int(worker.w), int(worker.h)
            
            # Body box (different colors for visual variety)
            body_color = (180, 100, 50) if worker.person_id % 2 == 0 else (50, 100, 180)
            cv2.rectangle(frame, (x, y), (x + w, y + h), body_color, -1)
            
            # Head circle
            head_cx = x + w // 2
            head_cy = y + h // 6 if not worker.is_fallen else y + h // 2
            head_r = 18
            cv2.circle(frame, (head_cx, head_cy), head_r, (200, 170, 150), -1)
            
            # Helmet (drawn on top of the head)
            if worker.has_helmet:
                # Draw a yellow semi-circle
                cv2.ellipse(frame, (head_cx, head_cy - 4), (16, 12), 0, 180, 360, (0, 220, 220), -1)
                # Helmet brim
                cv2.line(frame, (head_cx - 20, head_cy - 4), (head_cx + 20, head_cy - 4), (0, 220, 220), 3)

            # Glasses
            if worker.has_glasses:
                # Small blue spectacles
                cv2.circle(frame, (head_cx - 6, head_cy), 4, (255, 100, 0), 2)
                cv2.circle(frame, (head_cx + 6, head_cy), 4, (255, 100, 0), 2)
                cv2.line(frame, (head_cx - 2, head_cy), (head_cx + 2, head_cy), (255, 100, 0), 1)

            # Exposed skin (tattoos simulator)
            if worker.has_exposed_tattoo:
                # Draw arm patches on the sides
                cv2.rectangle(frame, (x - 4, y + 40), (x, y + 80), (140, 180, 140), -1)  # skin color with green tattoo ink

        # Overlay smoke particles if active
        if self.smoke_active:
            # Generate new smoke particles
            if len(self.smoke_particles) < 20:
                self.smoke_particles.append({
                    "x": random.randint(0, self.width),
                    "y": self.height + 20,
                    "r": random.randint(30, 80),
                    "alpha": random.uniform(0.1, 0.4)
                })
            
            # Create smoke overlay layer
            smoke_layer = frame.copy()
            for p in self.smoke_particles:
                p["y"] -= 2  # float up
                p["x"] += random.choice([-1, 0, 1])
                cv2.circle(smoke_layer, (p["x"], p["y"]), p["r"], (120, 120, 120), -1)
            
            # Blend
            cv2.addWeighted(smoke_layer, 0.25, frame, 0.75, 0, frame)
            # Filter out top particles
            self.smoke_particles = [p for p in self.smoke_particles if p["y"] > -50]

        return frame

    def next_frame(self) -> FrameData:
        self.frame_index += 1
        self.update_simulation()
        
        raw_frame = self.draw_scene()
        
        # Convert simulator persons to TrackedPerson structures
        tracked_persons = []
        for worker in self.active_sims:
            xmin, ymin, w, h = int(worker.x), int(worker.y), int(worker.w), int(worker.h)
            
            person = TrackedPerson(
                person_id=worker.person_id,
                bbox=[xmin, ymin, xmin + w, ymin + h],
                confidence=0.95,
                has_helmet=worker.has_helmet,
                has_glasses=worker.has_glasses,
                is_fallen=worker.is_fallen
            )
            person.metadata["has_exposed_tattoo"] = worker.has_exposed_tattoo
            tracked_persons.append(person)
            
        # Create pipeline package
        data = FrameData(
            frame_index=self.frame_index,
            timestamp=time.time(),
            raw_frame=raw_frame,
            processed_frame=raw_frame.copy(),
            persons=tracked_persons,
            is_smoke_detected=self.smoke_active
        )
        
        return data
