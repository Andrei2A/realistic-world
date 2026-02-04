import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';
import { OrbitControls } from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';

// 1. МИР
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x222222); // Темно-серый фон, чтобы видеть тени

// 2. КАМЕРА
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(2, 2, 5); // Поставили камеру сбоку и сверху
camera.lookAt(0, 0, 0); // Смотрим в центр мира

// 3. РЕНДЕРЕР (Твое 4К)
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true; // РАЗРЕШАЕМ ТЕНИ (для реализма)
document.body.appendChild(renderer.domElement);

// УПРАВЛЕНИЕ КАМЕРОЙ (Крути мышкой!)
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; // Плавное торможение
controls.dampingFactor = 0.05;

// 4. СВЕТ (Вот без этого был черный экран!)
const light = new THREE.DirectionalLight(0xffffff, 2); // Мощное солнце
light.position.set(5, 10, 7);
light.castShadow = true; // Солнце создает тени
scene.add(light);

const ambient = new THREE.AmbientLight(0xffffff, 0.3); // Мягкая подсветка со всех сторон
scene.add(ambient);

// 5. ПЕРВЫЙ РЕАЛИСТИЧНЫЙ БЛОК
const geometry = new THREE.BoxGeometry(1, 1, 1);
// Используем MeshStandardMaterial — он "дружит" с видеокартами RTX
const material = new THREE.MeshStandardMaterial({
    color: 0x00ff00,
    roughness: 0.3, // Степень шершавости
    metalness: 0.2  // Небольшой металлический блеск
});
const cube = new THREE.Mesh(geometry, material);
cube.castShadow = true; // Блок отбрасывает тень
scene.add(cube);

// 6. ПОЛ (Чтобы тени было на чем видеть)
const planeGeo = new THREE.PlaneGeometry(10, 10);
const planeMat = new THREE.MeshStandardMaterial({ color: 0x444444 });
const floor = new THREE.Mesh(planeGeo, planeMat);
floor.rotation.x = -Math.PI / 2;
floor.position.y = -1;
floor.receiveShadow = true; // На пол падает тень
scene.add(floor);

// АДАПТАЦИЯ ПОД РАЗМЕР ОКНА
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// 7. АНИМАЦИЯ
function animate() {
    requestAnimationFrame(animate);
    cube.rotation.y += 0.01; // Кубик крутится, чтобы видеть блики
    controls.update(); // Обновляем управление
    renderer.render(scene, camera);
}
animate();
