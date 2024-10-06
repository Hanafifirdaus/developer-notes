---
description: >-
  Penjelasan membuat toggle expander pada React tanpa menggunakan useState untuk
  mengurangi kerja javascript.
---

# Expander Pada React Menggunakan useState

_Toggle expander_ biasanya adalah component yang dibuat untuk membuat konent yang dapat kita tutup (_collapsible_) dan memunculkannya kembali (_expandable_) dengan mengklik suatu toggle. Cara untuk membuat expander tanpa menggunakan state agar dapat mengurangi proses _rerender_ adalah dengan menggunakan useRef. Untuk mempermudah memberikan css kita juga akan menggunakan tailwind.

Pertama kita membuat sebuah checkbox untuk kita jadikan sebagai _toggle_, di mana _checkbox_ ini ketika dia dalam posisi _checked_ akan membuka _content_ kita sementara sebaliknya akan menutup _content_ kita. Lalu kita masukan _element input checkbox_ tersebut kepada useRef. Karena element ini hanya sebagai toggle dan bukan interaksi langsung dengan user maka kita tambahkan kelas _hidden_ agar user tidak dapat melihatnya.

```javascript
import React, { useRef } from 'react';

const ExpandAbleComponent = () => {
  const checkboxRef = useRef(null);
  return (
    <input 
      type="checkbox" 
      ref={checkboxRef} 
      className="hidden" 
    />
  );
};

export default ExpandAbleComponent;
```

Selanjutnya kita akan menambahkan suatu toggle yang berfungsi untuk mengubah _checkbox_ menjadi _checked_ atau sebaliknya. Kali ini kita akan membuatnya dalam bentuk element button. Lalu pada _listener_ onClick kita menambahkan ref tadi lalu kita trigger _click_. Dengan begini kita sudah bisa mengubah state checkbox.

```javascript
import React, { useRef } from 'react';

const ExpandAbleComponent = () => {
  const checkboxRef = useRef(null);

  const handleButtonClick = () => {
    if (!checkboxRef.current) return;
    checkboxRef.current.click();
  };

  return (
    <>
      <button 
        className="bg-blue-500 text-white px-4 py-2 mt-4 rounded"
        onClick={handleButtonClick}
      >
        Toggle Checkbox
      </button>

      <input 
        type="checkbox" 
        ref={checkboxRef} 
        className="hidden" 
      />
    </>
  );
};

export default ExpandAbleComponent;
```

Lalu kita akan menambahkan content yang akan terbuka saat _checkbox_ kita sedang dalam posisi _checked_ dan sebaliknya. Untuk contoh sederhana kita akan menggunakan overflow-hidden agar _content_ tidak terlihat, transition untuk memberikan animasi ketika buka tutup dan min-h-0 untuk state saat tertutup.

```javascript
import React, { useRef } from 'react';

const ExpandAbleComponent = () => {
  const checkboxRef = useRef(null);

  const handleButtonClick = () => {
    if (!checkboxRef.current) return;
    checkboxRef.current.click();
  };

  return (
    <>
      <button 
        className="bg-blue-500 text-white px-4 py-2 mt-4 rounded"
        onClick={handleButtonClick}
      >
        Toggle Checkbox
      </button>

      <input 
        type="checkbox" 
        ref={checkboxRef} 
        className="hidden" 
      />
      <div className="min-h-0 overflow-hidden transition-all duration-300 ease-in-out">
        <p>This is a collapsible/expandable component.</p>
      </div>
    </>
  );
};

export default ExpandAbleComponent;
```

Terakhir kita akan menggunakan fitur tailwind yang bernama [_peer_](https://tailwindcss.com/docs/hover-focus-and-other-states#styling-based-on-sibling-state). Fungsi dari fitur ini adalah kita dapat memodifikasi style element berdasarkan element dari saudara sebelumnya. Dengan fitur ini kita dapat memasang [_peer_](https://tailwindcss.com/docs/hover-focus-and-other-states#styling-based-on-sibling-state) ke pada _checkbox_ kita dan kita dapat mengatur style untuk sodara _checkbox_ setelahnya. Dalam kasus kita itu berarti kita dapat mengubah style content kita.

```javascript
import React, { useRef } from 'react';

const ExpandAbleComponent = () => {
  const checkboxRef = useRef(null);

  const handleButtonClick = () => {
    if (!checkboxRef.current) return;
    checkboxRef.current.click();
  };

  return (
    <>
      <button 
        className="bg-blue-500 text-white px-4 py-2 mt-4 rounded"
        onClick={handleButtonClick}
      >
        Toggle Checkbox
      </button>

      <input 
        type="checkbox" 
        ref={checkboxRef} 
        className="hidden peer" 
      />
      <div className="min-h-0 peer-checked:min-h-[500px] overflow-hidden transition-all duration-300 ease-in-out">
        <p>This is a collapsible/expandable component.</p>
      </div>
    </>
  );
};

export default ExpandAbleComponent;
```

Dengan menambahkan [_peer_](https://tailwindcss.com/docs/hover-focus-and-other-states#styling-based-on-sibling-state) kita menambahkan juga class peer-checked:min-h-\[500px]. Pada tailwind, walaupun sudah terdapat min-h-0, saat kita menambahkan peer-checked:min-h-\[500px], style yang akan muncul adalah min-h-\[500px].

Itulah cara kita membuat _toggle expander_ pada react tanpa menggunakan useState. Dengan menggunakan cara ini kita dapat meminimalisir proses javascript dan membuat halaman kita lebih cepat.
