(() => {
    function qs(root, sel) {
        return root.querySelector(sel);
    }

    function qsa(root, sel) {
        return Array.from(root.querySelectorAll(sel));
    }

    function parseFloatSafe(v) {
        if (v === null || v === undefined) return null;
        const s = String(v).trim();
        if (!s) return null;
        const n = Number(s);
        return Number.isFinite(n) ? n : null;
    }

    function is01(v) {
        const s = String(v).trim();
        return s === "0" || s === "1";
    }

    function mpgValid(v) {
        const s = String(v).trim();
        if (!s) return true;
        return /^\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?$/.test(s);
    }

    function setErrorBox(errorBox, messages) {
        if (!errorBox) return;
        if (!messages.length) {
            errorBox.style.display = "none";
            errorBox.innerHTML = "";
            return;
        }
        errorBox.style.display = "block";
        errorBox.innerHTML = "<b>Vui lòng kiểm tra:</b><ul style='margin:8px 0 0 18px'>" +
            messages.map(m => `<li>${m}</li>`).join("") +
            "</ul>";
    }

    function renumberCards(container) {
        const cards = qsa(container, ".car-card");
        cards.forEach((card, idx) => {
            const title = qs(card, ".car-title");
            if (title) title.textContent = `Xe #${idx + 1}`;
        });
    }

    function getNextIndex(container) {
        const cards = qsa(container, ".car-card");
        let maxIdx = -1;
        for (const c of cards) {
            const i = Number(c.getAttribute("data-index"));
            if (Number.isFinite(i)) maxIdx = Math.max(maxIdx, i);
        }
        return maxIdx + 1;
    }

    function addCar(container) {
        const template = qs(container, "template");
        const list = qs(container, ".car-list");
        if (!template || !list) return;

        const idx = getNextIndex(container);
        const html = template.innerHTML.replaceAll("__INDEX__", String(idx));
        const wrapper = document.createElement("div");
        wrapper.innerHTML = html.trim();
        const card = wrapper.firstElementChild;
        if (!card) return;

        list.appendChild(card);
        renumberCards(container);

        return card;
    }

    function removeCar(container, card) {
        const minCars = Number(container.getAttribute("data-min")) || 1;
        const list = qs(container, ".car-list");
        if (!list) return;

        const cards = qsa(list, ".car-card");
        if (cards.length <= minCars) return;

        card.remove();
        renumberCards(container);
    }

    function anyFieldFilled(card) {
        const inputs = qsa(card, "input");
        return inputs.some(i => String(i.value || "").trim() !== "");
    }

    function validateForm(container, form) {
        const minCars = Number(container.getAttribute("data-min")) || 1;
        const errorBox = qs(form, ".client-errors");

        const list = qs(container, ".car-list");
        const cards = list ? qsa(list, ".car-card") : [];

        const usedCards = cards.filter(anyFieldFilled);
        const errors = [];

        if (usedCards.length < minCars) {
            errors.push(`Cần nhập tối thiểu ${minCars} xe (ít nhất 1 trường có giá trị).`);
        }

        usedCards.forEach((card, idx) => {
            const prefix = card.getAttribute("data-prefix") || "car";
            const index = card.getAttribute("data-index");
            const name = (field) => `${prefix}${index}_${field}`;
            const get = (field) => {
                const el = qs(card, `[name='${name(field)}']`);
                return el ? String(el.value || "").trim() : "";
            };

            const carNo = idx + 1;

            const price = get("price");
            const mileage = get("mileage");
            const year = get("year");
            const accidents = get("accidents_or_damage");
            const oneOwner = get("one_owner");
            const mpg = get("mpg");
            const driverRating = get("driver_rating");
            const sellerRating = get("seller_rating");
            const priceDrop = get("price_drop");

            const requiredMissing = [];
            if (!price) requiredMissing.push("price");
            if (!mileage) requiredMissing.push("mileage");
            if (!year) requiredMissing.push("year");
            if (!accidents) requiredMissing.push("accidents_or_damage");
            if (!oneOwner) requiredMissing.push("one_owner");

            if (requiredMissing.length) {
                errors.push(`Xe #${carNo}: thiếu trường bắt buộc: ${requiredMissing.join(", ")}.`);
            }

            const priceN = parseFloatSafe(price);
            if (price && (priceN === null || priceN < 0)) errors.push(`Xe #${carNo}: price phải là số >= 0.`);

            const mileageN = parseFloatSafe(mileage);
            if (mileage && (mileageN === null || mileageN < 0)) errors.push(`Xe #${carNo}: mileage phải là số >= 0.`);

            const yearN = parseFloatSafe(year);
            if (year && (yearN === null || yearN < 1980 || yearN > 2035)) errors.push(`Xe #${carNo}: year không hợp lệ (1980-2035).`);

            if (accidents && !is01(accidents)) errors.push(`Xe #${carNo}: accidents_or_damage chỉ nhận 0 hoặc 1.`);
            if (oneOwner && !is01(oneOwner)) errors.push(`Xe #${carNo}: one_owner chỉ nhận 0 hoặc 1.`);

            if (driverRating) {
                const n = parseFloatSafe(driverRating);
                if (n === null || n < 0 || n > 5) errors.push(`Xe #${carNo}: driver_rating phải trong khoảng 0-5.`);
            }

            if (sellerRating) {
                const n = parseFloatSafe(sellerRating);
                if (n === null || n < 0 || n > 5) errors.push(`Xe #${carNo}: seller_rating phải trong khoảng 0-5.`);
            }

            if (priceDrop) {
                const n = parseFloatSafe(priceDrop);
                if (n === null || n < 0) errors.push(`Xe #${carNo}: price_drop phải là số >= 0.`);
            }

            if (!mpgValid(mpg)) {
                errors.push(`Xe #${carNo}: mpg sai định dạng (vd: 30 hoặc 39-38).`);
            }
        });

        setErrorBox(errorBox, errors);
        return errors.length === 0;
    }

    function initCarForm(container) {
        const addBtn = qs(container, ".btn-add-car");
        const list = qs(container, ".car-list");
        if (!addBtn || !list) return;

        addBtn.addEventListener("click", () => addCar(container));

        container.addEventListener("click", (ev) => {
            const btn = ev.target.closest(".btn-remove-car");
            if (!btn) return;
            const card = btn.closest(".car-card");
            if (!card) return;
            removeCar(container, card);
        });

        const form = container.closest("form");
        if (form) {
            form.addEventListener("submit", (ev) => {
                if (!validateForm(container, form)) {
                    ev.preventDefault();
                    ev.stopPropagation();
                    const err = qs(form, ".client-errors");
                    if (err) err.scrollIntoView({ behavior: "smooth", block: "start" });
                }
            });
        }
    }

    async function fetchJson(url) {
        const res = await fetch(url, { headers: { "Accept": "application/json" } });
        if (!res.ok) {
            const text = await res.text().catch(() => "");
            throw new Error(`HTTP ${res.status} ${text}`);
        }
        return await res.json();
    }

    function normalize01Js(v) {
        if (v === null || v === undefined) return "";
        if (typeof v === "boolean") return v ? "1" : "0";
        const s = String(v).trim().toLowerCase();
        if (!s) return "";
        if (s === "1" || s === "1.0") return "1";
        if (s === "0" || s === "0.0") return "0";
        if (s === "true" || s === "t" || s === "yes" || s === "y") return "1";
        if (s === "false" || s === "f" || s === "no" || s === "n") return "0";
        const n = Number(s);
        if (Number.isFinite(n)) {
            if (Math.abs(n - 1) < 1e-9) return "1";
            if (Math.abs(n - 0) < 1e-9) return "0";
        }
        return s;
    }

    function fillCardFromCar(card, car) {
        const prefix = card.getAttribute("data-prefix") || "car";
        const index = card.getAttribute("data-index");
        const set = (field, value) => {
            const el = qs(card, `[name='${prefix}${index}_${field}']`);
            if (!el) return;
            el.value = value === null || value === undefined ? "" : String(value);
        };

        set("price", car.price ?? "");
        set("mileage", car.mileage ?? "");
        set("year", car.year ?? "");
        set("accidents_or_damage", normalize01Js(car.accidents_or_damage));
        set("one_owner", normalize01Js(car.one_owner));
        set("mpg", car.mpg ?? "");
        set("driver_rating", car.driver_rating ?? "");
        set("seller_rating", car.seller_rating ?? "");
        set("price_drop", car.price_drop ?? "");
    }

    function getTargetCard(container) {
        const list = qs(container, ".car-list");
        const cards = list ? qsa(list, ".car-card") : [];
        const empty = cards.find(c => !anyFieldFilled(c));
        if (empty) return empty;
        const newCard = addCar(container);
        return newCard || (list ? list.lastElementChild : null);
    }

    function initSavedImport(container) {
        const box = qs(container, ".saved-import");
        if (!box) return;

        const select = qs(box, ".saved-car-select");
        const btn = qs(box, ".btn-import-saved");
        if (!select || !btn) return;

        let loaded = false;
        let loading = false;

        async function loadList() {
            if (loaded || loading) return;
            loading = true;
            try {
                const data = await fetchJson("/api/my-cars");
                const items = (data && data.items) ? data.items : [];
                select.innerHTML = "";
                const opt0 = document.createElement("option");
                opt0.value = "";
                opt0.textContent = "— Chọn xe —";
                select.appendChild(opt0);
                for (const it of items) {
                    const opt = document.createElement("option");
                    opt.value = String(it.id);
                    const meta = it.created_at ? ` • ${it.created_at}` : "";
                    opt.textContent = `${it.title || ('Xe #' + it.id)}${meta}`;
                    select.appendChild(opt);
                }
                loaded = true;
            } catch (e) {
                console.warn("Failed to load saved cars", e);
            } finally {
                loading = false;
            }
        }

        // load on focus to avoid unnecessary call
        select.addEventListener("focus", loadList);
        btn.addEventListener("click", async () => {
            await loadList();
            const id = String(select.value || "").trim();
            if (!id) return;
            try {
                const data = await fetchJson(`/api/my-cars/${encodeURIComponent(id)}`);
                const car = (data && data.car) ? data.car : null;
                if (!car) return;
                const target = getTargetCard(container);
                if (!target) return;
                fillCardFromCar(target, car);
            } catch (e) {
                console.warn("Failed to import saved car", e);
            }
        });
    }

    document.addEventListener("DOMContentLoaded", () => {
        qsa(document, "[data-car-form='1']").forEach((c) => {
            initCarForm(c);
            initSavedImport(c);
        });
    });
})();
