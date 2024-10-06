# Menggunakan CSS variable pada tailwind

Untuk menambahkan css variable pada tailwind kita bisa menambahkan langsung nama variable yang kita inginkan ke dalam class nya.

````html
```html
<div class="w-screen h-screen flex items-center justify-center">
  <div class="bg-lime-500 h-72 w-[--width]">
      Content
  </div>
</div>
```
````

Terlihat bahwa pada element di atas kita menggunakan **\[--width]** untuk menentukan nilai width berdasarkan css variable **--width**. Jika kita hover pada class tersebut maka akan muncul seperti di bawah ini.

<figure><img src="../../.gitbook/assets/image (2).png" alt=""><figcaption><p>css variable width</p></figcaption></figure>

```css
.w-\[--width\] {
  width: var(--width);
}
```

pada initial value karena kita belum menambahkan variable widh nya maka tampilannya akan seperti ini.

<figure><img src="../../.gitbook/assets/image (3).png" alt=""><figcaption><p>sebelum set variable css</p></figcaption></figure>

````html
```html
<div class="w-screen h-screen flex items-center justify-center">
  <div class="bg-lime-500 h-72 w-[--width]" style="--width:300px">
      Content
  </div>
</div>

```
````

Perhatikan elemen di atas. kita menambahkan style `--width:300px` sehingga variable tersebut sudah mempunyai value 300px. Dengan ini width yang kita tambahkan akan mendapat value 300px.&#x20;
