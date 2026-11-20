# The part of the library shared by all jumps: binding a function on its first call.
#
# Every function jumps through a slot that starts empty (.bss). The first call finds the slot empty,
# arrives here with the slot's number in r11, and gets the address from vf_fwd_resolve (dlsym on the
# 12.9 library, loaded on the first call of any function). The registers that carry arguments are
# kept, so the function is then entered as if it had been called directly: a jump, not a call, so
# that arguments on the stack, structures by value and variable argument lists pass unchanged.

# On entry r11 = slot number, rsp % 16 == 8 (after a call). Returns the function's address in r11 and
# keeps every register that carries an argument, and r10.
    .hidden vf_fwd_bind
    .type vf_fwd_bind,@function
vf_fwd_bind:
    sub $200,%rsp
    mov %rdi,0(%rsp)
    mov %rsi,8(%rsp)
    mov %rdx,16(%rsp)
    mov %rcx,24(%rsp)
    mov %r8,32(%rsp)
    mov %r9,40(%rsp)
    mov %rax,48(%rsp)
    mov %r10,56(%rsp)
    movdqu %xmm0,64(%rsp)
    movdqu %xmm1,80(%rsp)
    movdqu %xmm2,96(%rsp)
    movdqu %xmm3,112(%rsp)
    movdqu %xmm4,128(%rsp)
    movdqu %xmm5,144(%rsp)
    movdqu %xmm6,160(%rsp)
    movdqu %xmm7,176(%rsp)
    mov %r11,%rdi
    call vf_fwd_resolve
    mov %rax,%r11
    mov 0(%rsp),%rdi
    mov 8(%rsp),%rsi
    mov 16(%rsp),%rdx
    mov 24(%rsp),%rcx
    mov 32(%rsp),%r8
    mov 40(%rsp),%r9
    mov 48(%rsp),%rax
    mov 56(%rsp),%r10
    movdqu 64(%rsp),%xmm0
    movdqu 80(%rsp),%xmm1
    movdqu 96(%rsp),%xmm2
    movdqu 112(%rsp),%xmm3
    movdqu 128(%rsp),%xmm4
    movdqu 144(%rsp),%xmm5
    movdqu 160(%rsp),%xmm6
    movdqu 176(%rsp),%xmm7
    add $200,%rsp
    ret
    .size vf_fwd_bind,.-vf_fwd_bind

# Slow path of a plain jump, entered by a jump from the function's first instructions.
    .hidden vf_fwd_slow
    .type vf_fwd_slow,@function
vf_fwd_slow:
    sub $8,%rsp
    call vf_fwd_bind
    add $8,%rsp
    jmp *%r11
    .size vf_fwd_slow,.-vf_fwd_slow

# Functions whose returned status is renumbered (12.9's values to 13.5's). r11 = slot number, r10 =
# number of stack words to copy (even). Calls the 12.9 function with a copy of the caller's stack
# arguments (more words than it has is harmless: they are read, never written), then renumbers eax.
    .hidden vf_fwd_status_entry
    .type vf_fwd_status_entry,@function
vf_fwd_status_entry:
    push %rbp
    mov %rsp,%rbp
    push %rbx
    sub $8,%rsp
    lea vf_slots(%rip),%rax
    mov (%rax,%r11,8),%rax
    test %rax,%rax
    jnz 1f
    call vf_fwd_bind
    mov %r11,%rax
1:  mov %rax,%r11
    mov %r10,%rbx
    lea (,%rbx,8),%rax
    sub %rax,%rsp
    xor %eax,%eax
2:  cmp %rbx,%rax
    jae 3f
    mov 16(%rbp,%rax,8),%r10
    mov %r10,(%rsp,%rax,8)
    inc %rax
    jmp 2b
3:  call *%r11
    mov %eax,%edi
    call vf_fwd_xlat
    mov -8(%rbp),%rbx
    leave
    ret
    .size vf_fwd_status_entry,.-vf_fwd_status_entry
    .hidden vf_fwd_xlat
