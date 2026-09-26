const API_URL = "https://your-service.onrender.com/extract-map";

async function uploadBlueprint(fileInputEl, scene) {
  const file = fileInputEl.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(API_URL, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    console.error("extraction failed", res.status);
    return;
  }

  const data = await res.json();
  if (data.status !== "success") {
    console.error("extraction error", data);
    return;
  }

  buildWallsInScene(data.walls, scene);
}

function buildWallsInScene(walls, scene) {
  const wallHeight = 3;
  const material = new THREE.MeshStandardMaterial({ color: 0xcccccc });

  walls.forEach((wall) => {
    const [x1, y1] = wall.start;
    const [x2, y2] = wall.end;
    const length = Math.hypot(x2 - x1, y2 - y1);
    if (length < 0.01) return;

    const thickness = wall.thickness || 0.2;
    const geometry = new THREE.BoxGeometry(length, wallHeight, thickness);
    const mesh = new THREE.Mesh(geometry, material);

    const midX = (x1 + x2) / 2;
    const midY = (y1 + y2) / 2;
    const angle = Math.atan2(y2 - y1, x2 - x1);

    mesh.position.set(midX, wallHeight / 2, midY);
    mesh.rotation.y = -angle;

    scene.add(mesh);
  });
}

document.getElementById("blueprint-input").addEventListener("change", (e) => {
  uploadBlueprint(e.target, window.scene);
});
